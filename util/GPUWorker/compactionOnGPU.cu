//
// Created by lg on 2021/6/9.
//

#include "compactionOnGPU.cuh"
#include "GPUhelper.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define RESTART_INTERVAL 16

// Same as Footer::kEncodedLength
// Encoded length of a Footer.  Note that the serialization of a
// Footer will always occupy exactly this many bytes.  It consists
// of two block handles and a magic number.
static constexpr int FooterLength = 2 * (10 + 10) + 8;
static constexpr uint64_t kTableMagicNumber = 0xdb4775248b80fb57ull;
static constexpr int restartInterval = 16;

namespace gpu {
// after calling this function, numCumulate change from restart num to block num
__global__ void readBlockIndexOffset(const char* data, const uint32_t table_num,
                                     const u_offset* restart_end,
                                     uint32_t* numCumulate,
                                     const uint32_t* index_block_starts,
                                     uint32_t* blocks_index_offsets) {
  size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < table_num) {
    // read the restart point, from the end to the beginning.
    const char* data_ptr = data + restart_end[ix] - 4;
    const char* last_restart_offset = data_ptr;
    uint32_t index_block_pos = index_block_starts[ix];
    uint32_t* write_pos = blocks_index_offsets+ numCumulate[ix+1]-1;
    uint32_t* write_end = blocks_index_offsets+ numCumulate[ix];
    while (write_pos >= write_end) {
      *write_pos = GetFixed32(data_ptr) +index_block_pos;
      data_ptr -= 4;
      --write_pos;
    }
    // now we can check how many block indexes follow the last restart
    last_restart_offset =
        data + GetFixed32(last_restart_offset) + index_block_pos;
    data_ptr = data + GetFixed32(last_restart_offset - 4) + index_block_pos;

    int last_index_has_block = 0;
    uint32_t non_shared;
    uint32_t value_len;
    while (data_ptr < last_restart_offset) {
      ++data_ptr; //skip shared_bytes
      non_shared = reinterpret_cast<const uint8_t*>(data_ptr)[0];
      value_len = reinterpret_cast<const uint8_t*>(data_ptr)[1];
      if ( (non_shared | value_len) < 128 ) {
        // Fast path: all two values are encoded in one byte each
        data_ptr += 2;
      } else {
        GetVarinet32(data_ptr, &non_shared);
        GetVarinet32(data_ptr, &value_len);
      }

      data_ptr += non_shared+ value_len;
      ++last_index_has_block;
    }
    numCumulate[ix] =
        (numCumulate[ix] - 2*ix) * restartInterval + last_index_has_block;
  }
}

__global__ void readBlockHandle(const char* data,
                                const uint32_t* restart_offsets,
                                const uint32_t restart_num,
                                const uint32_t* block_num_cumulate,
                                uint64_t* data_block_offsets) {
  // TODO: we actually can further read data block and get record restart
  size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < restart_num) {
    const char * pos_ptr = data + restart_offsets[ix];
    const char* next_restart_pos = data + restart_offsets[ix+1];
    uint32_t non_shared, value_len;

    uint64_t* store_pos =
        data_block_offsets + block_num_cumulate[ix];

    while (pos_ptr < next_restart_pos) {
      ++pos_ptr; //skip shared_bytes since we don't care key.
      non_shared = reinterpret_cast<const uint8_t*>(pos_ptr)[0];
      value_len = reinterpret_cast<const uint8_t*>(pos_ptr)[1];
      if ( (non_shared | value_len) < 128 ) {
        // Fast path: all two values are encoded in one byte each
        pos_ptr += 2;
      } else {
        GetVarinet32(pos_ptr, &non_shared);
        GetVarinet32(pos_ptr, &value_len);
      }

      pos_ptr+= non_shared; // we don't care key
      GetVarint64(pos_ptr, store_pos, false); // block offset
      pos_ptr += value_len; // we don't need block size
      ++store_pos;
    }
  }
}

__global__ void getIndexBlockInfo(const char* data, const uint32_t table_num,
                             const u_offset* table_offset,
                             u_offset* block_info) {
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  // read footer
  // we don't add filter, so bypassing metaindex block
  if (ix < table_num) {
    const char* table_start = data + table_offset[ix];
    const char* cur_start = data + table_offset[ix + 1] - FooterLength;

    BlockHandle index_handle;
    // we don't add filter, so bypassing metaindex block
    GetVarint64(cur_start, &index_handle.offset_);
    ++cur_start;  // we know the size is 0
    // read index handle
    GetVarint64(cur_start, &index_handle.offset_);
    GetVarint64(cur_start, &index_handle.size_);

    // get restarts num
    cur_start = table_start + index_handle.offset_ + index_handle.size_ -
                sizeof(uint32_t);
    block_info[ix] = GetFixed32(cur_start);
    block_info[ix+table_num] = cur_start - data;  // restart_end
    block_info[ix+table_num*2] = table_start + index_handle.offset_ - data; // index block start offset

  }
}


void GPUCompactor::DoCompaction() {
  // I exploit Unified memory, the explicit transportation is unnecessary

  dim3 myBlock(4*32); // FIXME: magic number
  dim3 myGrid(uintCeil(input_table_num_, myBlock.x));
  int ui32_B = sizeof(uint32_t);

  uint32_t * blocks_info; // data restart num and index block handle for every sstable
  checkCudaErrors(cudaMallocManaged(&blocks_info,
                                    input_table_num_ * ui32_B * 3));

  getIndexBlockInfo<<<myGrid, myGrid>>>(input_data_, input_table_num_,
                                        input_table_offset_, blocks_info);

  checkCudaErrors(cudaGetLastError());

  // use CUB to scan the total num of restarts
  void *d_temp_storage = nullptr;
  uint32_t* numCumulate;
  size_t temp_storage_B = 0;
  checkCudaErrors(cudaMallocManaged(&numCumulate,
                    (input_table_num_ + 1) * ui32_B));
  numCumulate[0] = 0;
  checkCudaErrors(cub::DeviceScan::InclusiveSum(d_temp_storage,temp_storage_B,
                                                blocks_info, numCumulate +1,
                                                input_table_num_));
  checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_B));
  checkCudaErrors(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_B, blocks_info, numCumulate +1, input_table_num_));
  checkCudaErrors(cudaFree(d_temp_storage));

  uint32_t total_restart_num = numCumulate[input_table_num_] / restartInterval;
  uint32_t* restarts_ends = blocks_info + input_table_num_;
  uint32_t* index_block_starts = restarts_ends + input_table_num_;

  // The last restart of each table's index_block is just a marker(we reach the end)
  uint32_t useful_restart_num = total_restart_num - input_table_num_;

  // 1. restart of data block index position
  // 2. offset of the index block
  uint32_t* block_index_handle_offsets;
  checkCudaErrors(cudaMallocManaged(&block_index_handle_offsets, total_restart_num*ui32_B));
  readBlockIndexOffset<<<myGrid, myBlock>>>(
      input_data_, input_table_num_, restarts_ends, numCumulate+1,
      index_block_starts, block_index_handle_offsets);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaFree(blocks_info));

  uint64_t * data_blocks_poss; // be careful of this type

  uint32_t total_blocks_num = numCumulate[input_table_num_];
  checkCudaErrors(cudaMallocManaged(&data_blocks_poss, (total_blocks_num)*sizeof(uint64_t)));
  myBlock.x = 16 * 32; // FIXME: magic number
  myGrid.x = uintCeil(useful_restart_num, myBlock.x);

  // read data block offsets
  uint32_t* restart_offsets = block_index_handle_offsets;
  readBlockHandle<<<myGrid, myBlock>>>(input_data_, restart_offsets,
                                       useful_restart_num, numCumulate,
                                       data_blocks_poss);
  checkCudaErrors(cudaGetLastError());


  // compaction
//  GPUMerge(d_key_buffer, d_buffer_offset);  // in-place update??
  // pack table
  char* d_output_data{};
  int* d_output_table_offset{};
//  PackTable<<<>>>(d_key_buffer, d_key_value, &d_output_data,
//                  &d_output_table_offset);
  // the input and output data will be free when the object is destroyed
  checkCudaErrors(cudaFree(blocks_info));
}
void GPUCompactor::unifiedAlloc(void** pointer, size_t size) {
  checkCudaErrors(cudaMallocManaged(pointer, size, cudaMemAttachHost));
}
void GPUCompactor::setInputData(const char* inputData) {
  input_data_ = inputData;
}
void GPUCompactor::setInputLevelOffset(const u_offset* inputLevelOffset) {
  input_level_offset_ = inputLevelOffset;
}
void GPUCompactor::setInputTableOffset(const u_offset* inputTableOffset) {
  input_table_offset_ = inputTableOffset;
}
GPUCompactor::~GPUCompactor() {
  checkCudaErrors(cudaFree(output_data_));
  checkCudaErrors(cudaFree((void*)input_level_offset_));
  checkCudaErrors(cudaFree((void*)input_table_offset_));
}
void GPUCompactor::unpackTable() {
  ;
}
}  // namespace gpu
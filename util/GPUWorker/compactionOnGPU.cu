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
// after calling this function, restartNumCumulate change from restart num to block num
__global__ void readBlockIndexOffset(const char* data, const uint32_t table_num,
                                     const offset_t* restart_end,
                                     const amount_t* restartNumCumulate,
                                     const addr_t* index_block_starts,
                                     offset_t* blocks_index_offsets,
                                     amount_t* whichTables,
                                     amount_t* blockNumCumulate) {
  size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < table_num) {
    // read the restart point, from the end to the beginning.
    const char* data_ptr = data + restart_end[ix] - 4;
    const char* last_restart_offset = data_ptr - 4; // actually the second last
    addr_t index_block_pos = index_block_starts[ix];
    amount_t* tableIdx_ptr = whichTables + restartNumCumulate[ix];
    offset_t* write_end = blocks_index_offsets + restartNumCumulate[ix];
    offset_t* write_pos = blocks_index_offsets + restartNumCumulate[ix + 1] - 1;
    while (write_pos >= write_end) {
      *tableIdx_ptr = ix;
      *write_pos = GetFixed32(data_ptr) + index_block_pos;
      data_ptr -= 4;
      --write_pos;
    }
    // now we can check how many block indexes follow the last restart
    data_ptr = data + GetFixed32(last_restart_offset) + index_block_pos;

    int last_has_block = 0;
    uint32_t non_shared;
    uint32_t value_len;
    while (data_ptr < last_restart_offset) {
      ++data_ptr;  // skip shared_bytes
      non_shared = reinterpret_cast<const uint8_t*>(data_ptr)[0];
      value_len = reinterpret_cast<const uint8_t*>(data_ptr)[1];
      if ((non_shared | value_len) < 128) {
        // Fast path: all two values are encoded in one byte each
        data_ptr += 2;
      } else {
        GetVarinet32(data_ptr, &non_shared);
        GetVarinet32(data_ptr, &value_len);
      }

      data_ptr += non_shared + value_len;
      ++last_has_block;
    }
    // set restartNumCumulate to block amount accumulation
    blockNumCumulate[ix] =
        (restartNumCumulate[ix] - 2 * ix) * restartInterval + last_has_block;
  }
}

__global__ void readBlockHandle(const char* data,
                                const offset_t* restart_offsets,
                                const amount_t restart_num,
                                const amount_t* block_num_cumulate,
                                const amount_t* restartNumCumulate,
                                const amount_t* whichTables,
                                const addr_t * table_starts,
                                offset_t * data_block_ends,
                                amount_t* data_block_restart_num) {
  // TODO: we actually can further read data block and get record restart
  size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < restart_num) {
    const char * pos_ptr = data + restart_offsets[ix];
    const char* next_restart_pos = data + restart_offsets[ix+1];
    uint32_t non_shared, value_len;

    amount_t tableIdx = whichTables[ix];
    amount_t blocks_num_before = block_num_cumulate[tableIdx];
    amount_t rankInTable = ix - restartNumCumulate[tableIdx];

    offset_t* dataBlockEnd_pos = data_block_ends + blocks_num_before + rankInTable;
    amount_t* dataBlockNumRestart_pos =
        data_block_restart_num + blocks_num_before + rankInTable;
    addr_t block_addr;
    uint64_t block_size;
    const char* data_block_ptr;

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
      GetVarint64(pos_ptr, &block_addr);
      GetVarint64(pos_ptr, &block_size);

      // read block trailer
      data_block_ptr = data + table_starts[tableIdx] + block_addr - 4;
      *dataBlockNumRestart_pos = GetFixed32(data_block_ptr);
      *dataBlockEnd_pos = data_block_ptr - data;
      ++dataBlockNumRestart_pos;
      ++dataBlockEnd_pos;
    }
  }
}

__global__ void readIndexBlockInfo(const char* data, const amount_t table_num,
                                  const addr_t * table_offsets,
                                  uint32_t* restart_info,
                                  addr_t* block_starts) {
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  // read footer
  if (ix < table_num) {
    const char* table_start = data + table_offsets[ix];
    const char* cur_start = data + table_offsets[ix + 1] - FooterLength;

    BlockHandle index_handle;
    // we don't add filter, so bypassing metaindex block
    GetVarint64(cur_start, &index_handle.offset_);
    ++cur_start;  // we know the size is 0
    // read index handle
    GetVarint64(cur_start, &index_handle.offset_);
    GetVarint64(cur_start, &index_handle.size_);

    cur_start = table_start + index_handle.offset_ + index_handle.size_ - 4;
    restart_info[ix] = GetFixed32(cur_start); // restarts num
    restart_info[ix + table_num] = cur_start - data;  // restart_end
    block_starts[ix] =
        table_start + index_handle.offset_ - data;  // index block start offset
  }
}

__global__ void readRecordRestarts(const char* data, const amount_t block_num,
                                   const addr_t* block_offsets,
                                   addr_t* block_starts) {
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  // read footer without metaindex
  if (ix < block_num) {

  }
}

void GPUCompactor::DoCompaction() {

  unpackTable();

  // compaction
//  GPUMerge(d_key_buffer, d_buffer_offset);  // in-place update??

  // pack table
  char* d_output_data{};
  int* d_output_table_offset{};

//  PackTable<<<>>>(d_key_buffer, d_key_value, &d_output_data,
//                  &d_output_table_offset);
  // the input and output data will be free when the object is destroyed
}
void GPUCompactor::unifiedAlloc(void** pointer, size_t size) {
  checkCudaErrors(cudaMallocManaged(pointer, size, cudaMemAttachHost));
}
GPUCompactor::~GPUCompactor() {
  checkCudaErrors(cudaFree(output_data_));
  checkCudaErrors(cudaFree((void*)input_level_offset_));
  checkCudaErrors(cudaFree((void*)input_table_offset_));
}
void GPUCompactor::unpackTable() {
  // I exploit Unified memory, the explicit transportation is unnecessary

  dim3 myBlock(4*32); // FIXME: magic number
  dim3 myGrid(uintCeil(input_table_num_, myBlock.x));
  int sizeOfData = sizeof(uint32_t);
  int sizeOfAddr = sizeof(addr_t);

  uint32_t * restart_info; // data restart num and index block handle for every sstable
  checkCudaErrors(cudaMallocManaged(&restart_info,
                                    input_table_num_ * sizeOfData * 2));
  addr_t* block_starts;
  checkCudaErrors(cudaMallocManaged(&block_starts, input_table_num_*sizeOfAddr));

  readIndexBlockInfo<<<myGrid, myGrid>>>(input_data_, input_table_num_,
                                         input_table_offset_, restart_info,
                                         block_starts);

  checkCudaErrors(cudaGetLastError());

  // use CUB to scan the total num of restarts
  void *d_temp_storage = nullptr;
  amount_t* restartNumCumulate;
  size_t tempSize = 0;
  checkCudaErrors(cudaMallocManaged(&restartNumCumulate,
                                    (input_table_num_ + 1) * sizeOfData));
  restartNumCumulate[0] = 0;
  checkCudaErrors(cub::DeviceScan::InclusiveSum(d_temp_storage, tempSize,
                                                restart_info,
                                    restartNumCumulate +1,
                                                input_table_num_));
  checkCudaErrors(cudaMalloc(&d_temp_storage, tempSize));
  checkCudaErrors(cub::DeviceScan::InclusiveSum(d_temp_storage, tempSize,
                                                restart_info,
                                    restartNumCumulate +1, input_table_num_));
  checkCudaErrors(cudaFree(d_temp_storage));

  amount_t total_restart_num =
      restartNumCumulate[input_table_num_] / restartInterval;
  offset_t* restarts_ends = restart_info + input_table_num_;
  addr_t* index_block_starts = block_starts;

  // The last restart of each table's index_block is just a marker(we reach the end)
  amount_t useful_restart_num = total_restart_num - input_table_num_;

  // 1. restart of data block index position
  // 2. offset of the index block
  offset_t * blocks_index_offsets;
  amount_t * whichTables;
  amount_t * blockNumCumulate;
  checkCudaErrors(cudaMallocManaged(&blocks_index_offsets, total_restart_num* sizeOfData));
  checkCudaErrors(cudaMallocManaged(&whichTables, total_restart_num* sizeOfData));
  checkCudaErrors(cudaMallocManaged(&blockNumCumulate, total_restart_num* sizeOfData));
  readBlockIndexOffset<<<myGrid, myBlock>>>(
      input_data_, input_table_num_, restarts_ends, restartNumCumulate,
      index_block_starts, blocks_index_offsets, whichTables, blockNumCumulate);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaFree(restart_info));
  checkCudaErrors(cudaFree(block_starts));
  restart_info = nullptr;
  block_starts = nullptr;

  offset_t * data_blocks_pos;
  amount_t * dataBlockNumRestart;

  amount_t total_blocks_num = blockNumCumulate[input_table_num_];
  checkCudaErrors(cudaMallocManaged(&data_blocks_pos, total_blocks_num*sizeOfData));
  checkCudaErrors(cudaMallocManaged(&dataBlockNumRestart, total_blocks_num*sizeOfData));
  myBlock.x = 16 * 32; // FIXME: magic number
  myGrid.x = uintCeil(useful_restart_num, myBlock.x);

  // read data block trailer
  offset_t* restart_offsets = blocks_index_offsets;
  readBlockHandle<<<myGrid, myBlock>>>(
      input_data_, restart_offsets, useful_restart_num, blockNumCumulate,
      restartNumCumulate, whichTables, input_table_offset_, data_blocks_pos,
      dataBlockNumRestart);
  checkCudaErrors(cudaGetLastError());

  // read data block
//  readRecordRestarts
}
}  // namespace gpu
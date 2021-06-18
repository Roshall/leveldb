//
// Created by lg on 2021/6/9.
//

#include "compactionOnGPU.cuh"
#include "cuda_runtime.h"

namespace gpu {
void GPUCompactor::DoCompaction() {
//  char* d_data{};
//  u_offset* d_level_offset{}; //FIXME: can int hold offset?
//  u_offset* d_table_offset{};
//
//  u_offset data_bytes = sizeof(char) * d_level_offset[2];
//  u_offset level_offset_bytes = sizeof(int) * 3;
//  u_offset table_offset_bytes = sizeof(int) * (input_table_num_ + 1);
//  // alloc data in GPU
//  checkCudaErrors(cudaMalloc((void**)&d_data, data_bytes));
//  checkCudaErrors(cudaMalloc((void**)&d_level_offset, level_offset_bytes));
//  checkCudaErrors(cudaMalloc((void**)&d_table_offset, table_offset_bytes));
//  // transport data from main Mem to global mem
//  checkCudaErrors(
//      cudaMemcpy(d_data, input_data_, data_bytes, cudaMemcpyHostToDevice));
//  checkCudaErrors(cudaMemcpy(d_table_offset, input_table_offset_, table_offset_bytes, cudaMemcpyHostToDevice));
//  checkCudaErrors(cudaMemcpy(d_level_offset, input_level_offset_, level_offset_bytes, cudaMemcpyHostToDevice));
  // do compaction
  char* d_key_buffer{};  // can it be struct??
  char* d_key_value{};
  int* d_buffer_offset{};

//  UnpackTable<<<>>>(d_data, d_level_offset, d_table_offset, &d_key_value,
//                    &d_key_value, &d_buffer_offset);

  // compaction
//  GPUMerge(d_key_buffer, d_buffer_offset);  // in-place update??
  // pack table
  char* d_output_data{};
  int* d_output_table_offset{};
//  PackTable<<<>>>(d_key_buffer, d_key_value, &d_output_data,
//                  &d_output_table_offset);
  // TODO: move data back to memory
//  output_data_ = malloc(sizeof(char))
//  checkCudaErrors(cudaMemcpy(d_output_data, ))
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
}  // namespace gpu
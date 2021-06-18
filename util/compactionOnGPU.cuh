//
// Created by lg on 2021/6/9.
//

#ifndef LEVELDB_COMPACTIONONGPU_CUH
#define LEVELDB_COMPACTIONONGPU_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <db/dbformat.h>
//#include <leveldb/options.h>
#include <iostream>


typedef  unsigned int u_offset;

namespace gpu {
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

class GPUCompactor {
 public:
  GPUCompactor()= default;
  GPUCompactor(const char* inputData, const u_offset* inputLevelOffset,
               const u_offset* inputTableOffset, const int inputNum)
      : input_data_(inputData),
        input_level_offset_(inputLevelOffset),
        input_table_offset_(inputTableOffset),
        input_table_num_(inputNum) {}

  virtual ~GPUCompactor() {
    checkCudaErrors(cudaFree(output_data_));
    checkCudaErrors(cudaFree((void*)input_level_offset_));
    checkCudaErrors(cudaFree((void*)input_table_offset_));

    //    delete[] output_data_;
//    delete[] input_level_offset_;
//    delete[] input_table_offset_;
  }
  void DoCompaction();
  void setHasLevel0(bool hasLevel0) { has_level0 = hasLevel0; }
  void unifiedAlloc(void** pointer,size_t size);
  void setInputData(const char* inputData);
  void setInputLevelOffset(const u_offset* inputLevelOffset);
  void setInputTableOffset(const u_offset* inputTableOffset);

 private:
  const char* input_data_{};
  const u_offset* input_level_offset_{};
  const u_offset* input_table_offset_{};
  const int input_table_num_{};
  char* output_data_{};
  u_offset* output_table_offset_{};
  int output_num_{};
  char* smallest_and_largest_{};  // TODO: decide the format
  u_offset* s_and_l_offset_{};
  bool has_level0{};
  // TODO: add these two condition when checking
//  const leveldb::SequenceNumber smallest_snapshot_;
  // I need (int)block_restart_interval
//  const leveldb::Options* option_;
};
}

#endif  // LEVELDB_COMPACTIONONGPU_CUH

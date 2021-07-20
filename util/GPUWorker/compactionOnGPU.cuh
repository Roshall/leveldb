//
// Created by lg on 2021/6/9.
//

#ifndef LEVELDB_COMPACTIONONGPU_CUH
#define LEVELDB_COMPACTIONONGPU_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <db/dbformat.h>
//#include <leveldb/options.h>
#include <db/version_set.h>
#include <iostream>

using offset_t = uint32_t;
using addr_t = uint64_t;
using amount_t = uint32_t;

namespace gpu {

class GPUCompactor {
 public:

  GPUCompactor()= default;
  GPUCompactor(const char* inputData, const offset_t* inputLevelOffset,
               const addr_t * inputTableOffset, const int inputNum)
      : input_data_(inputData),
        input_level_offset_(inputLevelOffset),
        input_table_offset_(inputTableOffset),
        input_table_num_(inputNum) {}

  virtual ~GPUCompactor();
  void DoCompaction();
  void setHasLevel0(bool hasLevel0) { has_level0 = hasLevel0; }
  void unifiedAlloc(void** pointer,size_t size);

 private:
  const char* input_data_{};
  const offset_t* input_level_offset_{};
  const addr_t* input_table_offset_{};
  const int input_table_num_{};
  char* output_data_{};
  offset_t* output_table_offset_{};
  int output_num_{};
  char* smallest_and_largest_{};  // TODO: decide the format
  offset_t* s_and_l_offset_{};
  bool has_level0{};
  // TODO: add these two condition when checking
//  const leveldb::SequenceNumber smallest_snapshot_;
  // I need (int)block_restart_interval
//  const leveldb::Options* option_;

  // for unpacking table we need 5+ step:
  // 1. reach index block get maximum blocks num array
  // 2. read index block get block offset
  // 3. reach data block get maximum record num array
  // 4. read data block to obtain each key size and value offset
  // 5. read data block again for finally reading key
  void unpackTable();
  void compackTable();
  void packTable();

  friend bool leveldb::VersionSet::Prepare4GPU(
      Compaction* c, SequenceNumber smallest_snapshot);
};
}

#endif  // LEVELDB_COMPACTIONONGPU_CUH

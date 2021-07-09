//
// Created by lg on 2021/6/21.
//

#ifndef LEVELDB_GPUHELPER_CUH
#define LEVELDB_GPUHELPER_CUH
struct BlockHandle {
  uint64_t offset_{};
  uint64_t size_{};
};

__device__ bool GetVarint64(const char*& start, uint64_t* value, bool inplace=true) {
  //get value and move pointer
  uint64_t result = 0;
  uint32_t shift;
  const char* pch = start;
  for (shift = 0; shift <= 63; shift += 7) {
    uint64_t byte = *(reinterpret_cast<const uint8_t*>(pch));
    pch++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      break;
    }
  }

  if (shift > 63) {
    return false;
  } else {
    if (inplace) {
      start = pch;
    }
    return true;
  }
}

__device__ bool GetVarinet32(const char*& start, uint32_t* value) {
  uint32_t result = 0;
  uint32_t shift;
  for (shift = 0; shift <= 28; shift += 7) {
    uint32_t byte = *(reinterpret_cast<const uint8_t*>(start));
    start++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      break;
    }
  }

  if (shift > 28) {
    return false;
  } else {
    return true;
  }
}

__device__ uint32_t GetFixed32(const char* start) {
  const auto* buffer = reinterpret_cast<const uint8_t*>(start);
  return (static_cast<uint32_t>(buffer[0])) |
         (static_cast<uint32_t>(buffer[1]) << 8) |
         (static_cast<uint32_t>(buffer[2]) << 16) |
         (static_cast<uint32_t>(buffer[0]) << 24);
}

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

__device__ __host__ inline unsigned int uintCeil(unsigned int x, unsigned int y) { return (x-1)/y+1;}
#endif  // LEVELDB_GPUHELPER_CUH

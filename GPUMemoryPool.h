#ifndef GPU_MEMORY_POOL_H
#define GPU_MEMORY_POOL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <map>
#include "charm++.h"  // For CmiCreateLock, CmiLock, and CmiUnlock

// A very simple (singleton) GPU memory pool class for C++03.
class GPUMemoryPool {
public:
    static GPUMemoryPool& instance() {
        static GPUMemoryPool pool;
        return pool;
    }
    
    // Drop-in wrapper for cudaMalloc
    cudaError_t malloc(void** ptr, size_t size, const char* file, int line);
    
    // Drop-in wrapper for cudaFree
    cudaError_t free(void* ptr, const char* file, int line);
    
private:
    GPUMemoryPool();
    ~GPUMemoryPool();
    GPUMemoryPool(const GPUMemoryPool&) {}
    GPUMemoryPool& operator=(const GPUMemoryPool&) { return *this; }
    
    CmiNodeLock poolLock; // C++03 compliant lock from Charm++.
    // Map of free blocks keyed by their actual size.
    std::multimap<size_t, void*> freeBlocks;
    // Map to track allocated block sizes.
    std::map<void*, size_t> allocatedSizes;
};

// Templated inline wrapper to hide cast ugliness.
template <typename T>
inline cudaError_t gpuPoolMallocTyped(T** ptr, size_t size, const char* file, int line) {
    return GPUMemoryPool::instance().malloc(reinterpret_cast<void**>(ptr), size, file, line);
}

// Convenience macros to automatically use __FILE__ and __LINE__
#define gpuPoolMalloc(ptr, size) gpuPoolMallocTyped(ptr, size, __FILE__, __LINE__)
#define gpuPoolFree(ptr) GPUMemoryPool::instance().free(ptr, __FILE__, __LINE__)

#endif // GPU_MEMORY_POOL_H 
#include "GPUMemoryPool.h"

GPUMemoryPool::GPUMemoryPool() {
    poolLock = CmiCreateLock();
    printf("[GPUMemoryPool] Initialized memory pool.\n");
}

GPUMemoryPool::~GPUMemoryPool() {
    // Optionally, you might iterate and free all stored memory blocks here.
}

cudaError_t GPUMemoryPool::malloc(void** ptr, size_t size, const char* file, int line) {
    CmiLock(poolLock);
    
    // Attempt to find a free block that's at least the required size
    std::multimap<size_t, void*>::iterator it = freeBlocks.lower_bound(size);
    if (it != freeBlocks.end()) {
        *ptr = it->second;
        printf("[GPUMemoryPool] Reusing free block at %p of size %zu (requested %zu) [%s:%d]\n",
               *ptr, it->first, size, file, line);
        freeBlocks.erase(it);
        // Retrieve the recorded allocation size.
        size_t actualSize = allocatedSizes[*ptr];
        allocatedSizes.erase(*ptr);
        CmiUnlock(poolLock);
        return cudaSuccess;
    }
    
    // No suitable block found: fall back to cudaMalloc.
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaSuccess) {
        printf("[GPUMemoryPool] Allocated new block at %p of size %zu using cudaMalloc [%s:%d]\n",
               *ptr, size, file, line);
        // Record the allocated size.
        allocatedSizes[*ptr] = size;
    } else {
        printf("[GPUMemoryPool] cudaMalloc failed for size %zu at %s:%d\n", size, file, line);
    }
    CmiUnlock(poolLock);
    return err;
}

cudaError_t GPUMemoryPool::free(void* ptr, const char* file, int line) {
    CmiLock(poolLock);
    
    // Retrieve the actual block size from our tracking.
    size_t actualSize = 0;
    std::map<void*, size_t>::iterator it = allocatedSizes.find(ptr);
    if(it != allocatedSizes.end()) {
        actualSize = it->second;
    } else {
        // If not found, default to a conservative size.
        actualSize = 0;
    }
    
    printf("[GPUMemoryPool] Freeing block at %p for reuse (size %zu) [%s:%d]\n",
           ptr, actualSize, file, line);
    // Insert the block back into the pool with its actual size.
    freeBlocks.insert(std::make_pair(actualSize, ptr));
    
    CmiUnlock(poolLock);
    return cudaSuccess;
} 
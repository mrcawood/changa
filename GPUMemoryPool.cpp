#include "GPUMemoryPool.h"

GPUMemoryPool::GPUMemoryPool() {
    poolLock = CmiCreateLock();
    printf("[GPUMemoryPool] Initialized memory pool.\n");
}

GPUMemoryPool::~GPUMemoryPool() {
    // Optionally, release any remaining memory here.
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
        CmiUnlock(poolLock);
        return cudaSuccess;
    }
    
    // No suitable block found: fall back to cudaMalloc.
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaSuccess) {
        printf("[GPUMemoryPool] Allocated new block at %p of size %zu using cudaMalloc [%s:%d]\n",
               *ptr, size, file, line);
    } else {
        printf("[GPUMemoryPool] cudaMalloc failed for size %zu at %s:%d\n", size, file, line);
    }
    CmiUnlock(poolLock);
    return err;
}

cudaError_t GPUMemoryPool::free(void* ptr, const char* file, int line) {
    CmiLock(poolLock);
    
    // For this simple implementation, we simply add the pointer back to the pool with a dummy size.
    size_t dummySize = 0; // If desired, you can maintain an extra map pointer->size.
    printf("[GPUMemoryPool] Freeing block at %p for reuse [%s:%d]\n", ptr, file, line);
    freeBlocks.insert(std::make_pair(dummySize, ptr));
    
    CmiUnlock(poolLock);
    // Optionally, one could call cudaFree() if you want to release memory back to the OS.
    // For now, we keep the block alive in the pool.
    return cudaSuccess;
} 
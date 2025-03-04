#include "GPUMemoryPool.h"

GPUMemoryPool::GPUMemoryPool() {
    poolLock = CmiCreateLock();
    
    // Initialize analytics counters
    totalAllocations = 0;
    totalFrees = 0;
    totalReuses = 0;
    totalBytesAllocated = 0;
    totalMallocTime = 0.0;
    totalFreeTime = 0.0;
    
    printf("[GPUMemoryPool][PE %d] Initialized memory pool.\n", CmiMyPe());
}

GPUMemoryPool::~GPUMemoryPool() {
    // Print final analytics report
    printAnalyticsReport();
    
    // Free all stored memory blocks
    CmiLock(poolLock);
    
    for (std::multimap<size_t, void*>::iterator it = freeBlocks.begin(); 
         it != freeBlocks.end(); ++it) {
        cudaFree(it->second);
    }
    freeBlocks.clear();
    allocatedSizes.clear();
    
    CmiUnlock(poolLock);
    CmiDestroyLock(poolLock);
}

double GPUMemoryPool::getCurrentTimeSeconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

cudaError_t GPUMemoryPool::malloc(void** ptr, size_t size, const char* file, int line) {
    double startTime = getCurrentTimeSeconds();
    CmiLock(poolLock);
    
    // Attempt to find a free block that's at least the required size
    std::multimap<size_t, void*>::iterator it = freeBlocks.lower_bound(size);
    if (it != freeBlocks.end()) {
        *ptr = it->second;
        //printf("[GPUMemoryPool][PE %d] Reusing free block at %p of size %zu (requested %zu) [%s:%d]\n",
        //       CmiMyPe(), *ptr, it->first, size, file, line);
        freeBlocks.erase(it);
        
        // Update analytics
        totalReuses++;
        
        CmiUnlock(poolLock);
        totalMallocTime += (getCurrentTimeSeconds() - startTime);
        return cudaSuccess;
    }
    
    // No suitable block found: fall back to cudaMalloc.
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaSuccess) {
        printf("[GPUMemoryPool][PE %d] Allocated new block at %p of size %zu using cudaMalloc [%s:%d]\n",
               CmiMyPe(), *ptr, size, file, line);
        // Record the allocated size.
        allocatedSizes[*ptr] = size;
        
        // Update analytics
        totalAllocations++;
        totalBytesAllocated += size;
    } else {
        printf("[GPUMemoryPool][PE %d] cudaMalloc failed for size %zu at %s:%d\n", 
               CmiMyPe(), size, file, line);
    }
    CmiUnlock(poolLock);
    totalMallocTime += (getCurrentTimeSeconds() - startTime);
    return err;
}

cudaError_t GPUMemoryPool::free(void* ptr, const char* file, int line) {
    if (ptr == NULL) return cudaSuccess;
    
    double startTime = getCurrentTimeSeconds();
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
    
    //printf("[GPUMemoryPool] Freeing block at %p for reuse (size %zu) [%s:%d]\n",
    //       ptr, actualSize, file, line);
    // Insert the block back into the pool with its actual size.
    freeBlocks.insert(std::make_pair(actualSize, ptr));
    
    // Update analytics
    totalFrees++;
    
    CmiUnlock(poolLock);
    totalFreeTime += (getCurrentTimeSeconds() - startTime);
    return cudaSuccess;
}

void GPUMemoryPool::printAnalyticsReport() {
    CmiLock(poolLock);
    
    printf("\n==================================================\n");
    printf("GPUMemoryPool Analytics Report [PE %d]\n", CmiMyPe());
    printf("==================================================\n");
    printf("Total cudaMalloc allocations: %lu\n", totalAllocations);
    printf("Total memory blocks freed: %lu\n", totalFrees);
    printf("Total memory blocks reused: %lu\n", totalReuses);
    
    // Calculate average allocation size
    double avgAllocationSize = 0.0;
    if (totalAllocations > 0) {
        avgAllocationSize = (double)totalBytesAllocated / totalAllocations;
    }
    printf("Total bytes allocated: %lu bytes (%.2f MB)\n", 
           totalBytesAllocated, totalBytesAllocated / (1024.0 * 1024.0));
    printf("Average allocation size: %.2f bytes (%.2f KB)\n", 
           avgAllocationSize, avgAllocationSize / 1024.0);
    
    // Timing information
    printf("Total time spent in malloc operations: %.6f seconds\n", totalMallocTime);
    printf("Total time spent in free operations: %.6f seconds\n", totalFreeTime);
    printf("Total memory management time: %.6f seconds\n", totalMallocTime + totalFreeTime);
    
    // Current pool status
    printf("Current free blocks in pool: %lu\n", freeBlocks.size());
    
    // Calculate total memory currently in the pool
    size_t totalPoolMemory = 0;
    for (std::multimap<size_t, void*>::iterator it = freeBlocks.begin(); 
         it != freeBlocks.end(); ++it) {
        totalPoolMemory += it->first;
    }
    printf("Total memory in free pool: %lu bytes (%.2f MB)\n", 
           totalPoolMemory, totalPoolMemory / (1024.0 * 1024.0));
    
    printf("==================================================\n\n");
    
    CmiUnlock(poolLock);
} 

#ifndef GPUMEMORYPOOL_H
#define GPUMEMORYPOOL_H

#ifdef CUDA
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>

/**
 * @brief Represents a block of memory managed by the pool.
 */
struct PoolBlock {
    void* ptr;      // Pointer to the device memory
    size_t size;    // Size of the memory block
};

/**
 * @brief Represents a block that has been returned to the pool but is still
 *        in use by a CUDA stream.
 */
struct InFlightBlock {
    PoolBlock block;        // The memory block
    cudaEvent_t event;      // The CUDA event to track completion
};

/**
 * @brief Initializes the memory pool. Must be called once at startup before any
 *        pool operations are performed.
 */
void poolInit();

/**
 * @brief Allocates a block of memory from the pool for a specific stream.
 *        If no suitable block is found in the pool, a new one is allocated
 *        using cudaMalloc.
 * @param ptr Output pointer to the allocated device memory.
 * @param size The size of memory to allocate in bytes.
 * @param stream The CUDA stream that will use this memory.
 * @return cudaError_t result from the allocation operation.
 */
cudaError_t poolMalloc(void** ptr, size_t size, cudaStream_t stream);

/**
 * @brief Returns a block of memory to the pool. The memory is not immediately
 *        available for reuse; it becomes available only after all preceding
 *        work on the specified stream has completed.
 * @param ptr The device pointer to return to the pool.
 * @param stream The CUDA stream on which the memory was last used.
 * @return cudaError_t result, typically cudaSuccess.
 */
cudaError_t poolFree(void* ptr, cudaStream_t stream);

/**
 * @brief Destroys the memory pool, freeing all cached GPU memory. Must be called
 *        once at shutdown to prevent memory leaks.
 */
void poolDestroy();

// Forward declarations for the original tracked allocation functions
cudaError_t gpuMallocTracked(void** devPtr, size_t size, const char* tag, const char* functionTag, const char* file, int line);
cudaError_t gpuFreeTracked(void* devPtr, const char* tag, const char* functionTag, const char* file, int line);

// Templated inline wrapper to handle type casting safely and pass context
template <typename T>
inline cudaError_t gpuMallocTyped(T** ptr, size_t size, const char* pointerIdTag, const char* functionTag, const char* file, int line) {
    // Call the C-linkage function
    return gpuMallocTracked(reinterpret_cast<void**>(ptr), size, pointerIdTag, functionTag, file, line);
}

// Macro helper to call the typed wrapper, capturing context automatically
#define gpuMallocHelper(ptr, size, funcTag) gpuMallocTyped(ptr, size, #ptr, funcTag, __FILE__, __LINE__)

// Macro helper for gpuFree, capturing context automatically
#define gpuFreeHelper(ptr, funcTag) gpuFreeTracked(ptr, #ptr, funcTag, __FILE__, __LINE__)

#endif // CUDA

#endif // GPUMEMORYPOOL_H

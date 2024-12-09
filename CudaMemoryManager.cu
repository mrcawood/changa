#include "CudaMemoryManager.h"

/**
 * @brief Helper struct for gravity computation memory
 */
struct GravityMemory {
    void* d_list;
    void* d_bucketMarkers;
    void* d_bucketStarts;
    void* d_bucketSizes;
    size_t listSize;
    size_t markerSize;
    size_t startSize;
    size_t sizeSize;
};

/**
 * @brief Allocate memory needed for gravity computation
 * @param manager Memory manager instance
 * @param stream CUDA stream to use
 * @param prefix Unique prefix for this allocation set
 * @param listSize Size of interaction list
 * @param markerSize Size of bucket markers
 * @param startSize Size of bucket starts
 * @return GravityMemory struct with allocated pointers
 */
GravityMemory allocateGravityMemory(
    CudaMemoryManager& manager,
    cudaStream_t stream,
    const std::string& prefix,
    size_t listSize,
    size_t markerSize,
    size_t startSize
) {
    GravityMemory mem = {};
    
    // Store sizes
    mem.listSize = listSize;
    mem.markerSize = markerSize;
    mem.startSize = startSize;
    mem.sizeSize = startSize; // Same as startSize in current implementation

    // Allocate memory blocks
    mem.d_list = manager.allocate(listSize, prefix + "_list", stream);
    mem.d_bucketMarkers = manager.allocate(markerSize, prefix + "_markers", stream);
    mem.d_bucketStarts = manager.allocate(startSize, prefix + "_starts", stream);
    mem.d_bucketSizes = manager.allocate(mem.sizeSize, prefix + "_sizes", stream);

    return mem;
}

/**
 * @brief Transfer gravity computation data to device
 * @param manager Memory manager instance
 * @param mem GravityMemory struct with allocated memory
 * @param stream CUDA stream to use
 * @param h_list Host interaction list
 * @param h_markers Host bucket markers
 * @param h_starts Host bucket starts
 * @param h_sizes Host bucket sizes
 * @return True if all transfers initiated successfully
 */
bool transferGravityData(
    CudaMemoryManager& manager,
    const GravityMemory& mem,
    cudaStream_t stream,
    const void* h_list,
    const void* h_markers,
    const void* h_starts,
    const void* h_sizes
) {
    bool success = true;
    
    success &= manager.asyncTransfer(mem.d_list, h_list, mem.listSize, stream);
    success &= manager.asyncTransfer(mem.d_bucketMarkers, h_markers, mem.markerSize, stream);
    success &= manager.asyncTransfer(mem.d_bucketStarts, h_starts, mem.startSize, stream);
    success &= manager.asyncTransfer(mem.d_bucketSizes, h_sizes, mem.sizeSize, stream);

    return success;
}

/**
 * @brief Free gravity computation memory
 * @param manager Memory manager instance
 * @param prefix Prefix used for allocation
 */
void freeGravityMemory(
    CudaMemoryManager& manager,
    const std::string& prefix
) {
    manager.free(prefix + "_list");
    manager.free(prefix + "_markers");
    manager.free(prefix + "_starts");
    manager.free(prefix + "_sizes");
}

void someCriticalFunction() {
    // ... existing code ...
    manager.writeLog("Starting critical function execution");
    // ... critical operations ...
    manager.writeLog("Finished critical function execution");
    // ... existing code ...
} 
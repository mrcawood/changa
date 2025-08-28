#ifdef CUDA

#include "ParallelGravity.h"
#include "GPUMemoryPool.h" // Header for this file's functions/macros
#include "DataManager.h"
#include "memlog.h"
#include "ParallelGravity.decl.h"
#include <vector>
#include <unordered_map>

/**
 * @brief Wrapper for cudaMalloc that logs allocation events for memory tracking.
 * @param devPtr Pointer to allocated device memory (output)
 * @param size Size of memory to allocate in bytes
 * @param tag Identifier for the memory allocation (typically variable name)
 * @param functionTag Name of the calling function for context
 * @param file Source file where allocation occurs
 * @param line Line number where allocation occurs
 * @return cudaError_t result from cudaMalloc
 */
cudaError_t gpuMallocTracked(void** devPtr, size_t size, const char* tag, const char* functionTag, const char* file, int line) {

    double timestamp = CkWallTimer(); // Get timestamp before allocation
    cudaError_t result = cudaMalloc(devPtr, size);
    double timestamp_after = CkWallTimer(); // Get timestamp after allocation completes

    // Access node-local DataManager
    DataManager* dm = (DataManager*)CkLocalNodeBranch(dataManagerID);
    if (dm && dm->memLog && dm->bGpuMemLogger) { // Check if dm, memLog are valid and logging is enabled
        // Determine operation type based on allocation success/failure
        MemLogOpType opType = (result == cudaSuccess) ? MEMLOG_ALLOC : MEMLOG_ALLOC_FAIL;
        // Get allocated address if successful, otherwise record 0
        uintptr_t address = (result == cudaSuccess && *devPtr != NULL) ? (uintptr_t)(*devPtr) : 0;
        MemLogEvent event(CkMyNode(), opType, size, address, timestamp_after, file, line, tag, functionTag);

        CmiLock(dm->lockMemLog);
        dm->memLog->meTab.push_back(event);
        CmiUnlock(dm->lockMemLog);
    }

    return result;
}

/**
 * @brief Wrapper for cudaFree that logs deallocation events for memory tracking.
 * @param devPtr Pointer to device memory to free
 * @param tag Identifier for the memory deallocation (typically variable name)
 * @param functionTag Name of the calling function for context
 * @param file Source file where deallocation occurs
 * @param line Line number where deallocation occurs
 * @return cudaError_t result from cudaFree (cudaSuccess for NULL pointer)
 */
cudaError_t gpuFreeTracked(void* devPtr, const char* tag, const char* functionTag, const char* file, int line) {
    double timestamp = CkWallTimer();
    MemLogOpType opType;
    uintptr_t address = (uintptr_t)devPtr;

    // Access node-local DataManager
    DataManager* dm = (DataManager*)CkLocalNodeBranch(dataManagerID);

    if (devPtr == NULL) {
        // NULL pointer free is a no-op in CUDA, log as skipped operation
        opType = MEMLOG_FREE_SKIP;
        if (dm && dm->memLog && dm->bGpuMemLogger) {
             MemLogEvent event(CkMyNode(), opType, 0, address, timestamp, file, line, tag, functionTag);
             CmiLock(dm->lockMemLog);
             dm->memLog->meTab.push_back(event);
             CmiUnlock(dm->lockMemLog);
        }
        return cudaSuccess; 
    }

    cudaError_t result = cudaFree(devPtr);
    double timestamp_after = CkWallTimer();

    // Determine operation type based on free success/failure
    opType = (result == cudaSuccess) ? MEMLOG_FREE : MEMLOG_FREE_FAIL;

    if (dm && dm->memLog && dm->bGpuMemLogger) {
        MemLogEvent event(CkMyNode(), opType, 0, address, timestamp_after, file, line, tag, functionTag);
        CmiLock(dm->lockMemLog);
        dm->memLog->meTab.push_back(event);
        CmiUnlock(dm->lockMemLog);
    }

    return result;
}

// Implementation for the DataManager entry method to set the log filename
// This follows the starlog pattern where the implementation is in the feature's file.
void DataManager::initMemLog(std::string _fileName, int bGpuMemLoggerFlag, const CkCallback &cb) {
    CmiLock(lockMemLog);
    if (memLog != nullptr) { 
        memLog->fileName = _fileName;
    } else {
        CkPrintf("WARNING PE %d: memLog is NULL in initMemLog! Cannot set filename.\n", CkMyPe());
    }
    bGpuMemLogger = bGpuMemLoggerFlag; // Set the logging flag
    CmiUnlock(lockMemLog);
    // Signal completion for this PE in the collective operation
    contribute(cb);
}

/// @brief Initializes the memory log file on PE 0 and sends filename to all DataManagers.
/// Mimics the Main::initStarLog pattern.
void Main::initMemLog() {
    std::string memLogFile = "memlog.out"; // Hardcoded filename

    // Send filename and flag to all DataManagers and wait for completion
    // Call the init function on the DataManager on all PEs
    dMProxy.initMemLog(memLogFile, param.bGpuMemLogger, CkCallbackResumeThread());
    // Implicit wait for all PEs to finish DataManager::initMemLog happens here

    // PE 0 creates/truncates the file and writes a header AFTER collective operation completes
    if (CkMyPe() == 0) {
        FILE* fpLog = CmiFopen(memLogFile.c_str(), "w");
        fprintf(fpLog, "# ChaNGa Memory Log v1.1\n");
        fprintf(fpLog, "# NodeID OpType Size Address Timestamp File:Line PointerID FunctionTag\n");
        int close_err = CmiFclose(fpLog);
        if (close_err != 0) {
            CkPrintf("WARNING: PE 0 failed to close memlog file: %s (Error %d)\n", memLogFile.c_str(), close_err);
        }
    }
}

/// @brief Flush memlog table to disk sequentially across nodes.
/// This coordinates the flush; the actual writing happens in MemLog::flush().
void DataManager::flushMemLog(const CkCallback& cb) {
    // Call the actual file writing implementation in MemLog
    // Assumes memLog->flush() handles checking if the buffer is empty,
    // file opening/writing/closing, error checking, and buffer clearing.
    if (memLog) { // Ensure memLog is not null
         memLog->flush();
    } else {
         // Log a warning if memLog is unexpectedly null
         // Use CkPrintf for Charm++ compatible output
         CkPrintf("WARNING Node %d: memLog is NULL in flushMemLog! Skipping flush.\n", thisIndex);
    }

    // Sequential node flushing: ensures ordered writes to avoid file corruption
    if(thisIndex != CkNumNodes()-1) {
        // Pass the call to the next node, forwarding the final callback
        thisProxy[thisIndex + 1].flushMemLog(cb);
    } else {
        // We are the last node, signal completion of the entire sequence
        cb.send();
    }
}

/// @brief Flush buffered memory log events to the designated file.
/// This function performs the actual file I/O for the memory log.
void MemLog::flush() {
    if (meTab.empty()) {
        return; // Nothing to flush
    }

    FILE* outfile = CmiFopen(fileName.c_str(), "a"); // Open in append mode

    if (outfile == NULL) {
        // Use CkPrintf for Charm++ compatible error output. Avoid aborting for logging failures.
        CkPrintf("WARNING: Could not open memlog file '%s' for appending.\n", fileName.c_str());
        return; 
    }

    // Iterate through the buffered events and write them to the file
    for (const auto& event : meTab) {
        const char* opTypeStr;
        switch (event.opType) {
            case MEMLOG_ALLOC:      opTypeStr = "ALLOC";      break;
            case MEMLOG_FREE:       opTypeStr = "FREE ";      break; // Padded for alignment
            case MEMLOG_ALLOC_FAIL: opTypeStr = "ALLOC_F";    break;
            case MEMLOG_FREE_FAIL:  opTypeStr = "FREE_F ";    break;
            case MEMLOG_FREE_SKIP:  opTypeStr = "FREE_S ";    break;
            case MEMLOG_POOL_REUSE:   opTypeStr = "REUSE";    break;
            case MEMLOG_POOL_RELEASE: opTypeStr = "RELEASE";  break;
            default:                opTypeStr = "UNKNOWN";    break;
        }

        // Format: NodeID OpType Size Address Timestamp File:Line PointerID FunctionTag
        // Use %d for NodeID, %zu for size_t, %p for pointer (address), %.6f for timestamp
        // Assumes strings do not contain problematic characters (spaces, newlines)
        fprintf(outfile, "%d %s %zu %p %.6f %s %s %s\n",
                event.nodeId,
                opTypeStr,
                event.size,
                (void*)event.address, // Cast uintptr_t back to void* for %p
                event.timestamp,
                event.location.c_str(),
                event.pointerId.c_str(),
                event.functionTag.c_str());
    }

    int result = CmiFclose(outfile);
    if (result != 0) {
        CkPrintf("WARNING: Failed to close memlog file '%s' properly (Error %d).\n", fileName.c_str(), result);
        // Continue even if close fails, data might still be flushed
    }

    // Clear the buffer now that events are written
    meTab.clear();
}

// Tiered bucket system for memory pool - optimized for astronomy simulations
static size_t getBucketSize(size_t requestedSize) {
    // Tiered bucket system optimized for astronomy data:
    // Small: <= 1MB in 64KB increments (for small interaction lists)
    // Medium: 1MB-100MB in 1MB increments (for typical particle arrays)  
    // Large: 100MB-1GB in 10MB increments (for large moment arrays)
    // XLarge: >1GB in 100MB increments (for massive datasets)
    
    if (requestedSize <= 1024 * 1024) { // <= 1MB
        // Round up to nearest 64KB
        return ((requestedSize + 65535) / 65536) * 65536;
    } else if (requestedSize <= 100 * 1024 * 1024) { // <= 100MB
        // Round up to nearest 1MB
        return ((requestedSize + 1048575) / 1048576) * 1048576;
    } else if (requestedSize <= 1024 * 1024 * 1024) { // <= 1GB
        // Round up to nearest 10MB
        return ((requestedSize + 10485759) / 10485760) * 10485760;
    } else { // > 1GB
        // Round up to nearest 100MB
        return ((requestedSize + 104857599) / 104857600) * 104857600;
    }
}

// Core data structures for the memory pool
static CmiNodeLock g_poolLock = nullptr; // Node-level lock for thread safety
static std::unordered_map<size_t, std::vector<PoolBlock>> g_free_pool;
static std::vector<InFlightBlock> g_in_flight_blocks;
static std::unordered_map<void*, size_t> g_pointer_sizes;

/**
 * @brief Private helper to reclaim blocks from the in-flight list whose CUDA
 *        events have completed. This function must be called inside a lock.
 */
static void reclaim_completed_blocks() {
    // Using an index-based loop to safely remove elements while iterating
    for (size_t i = 0; i < g_in_flight_blocks.size(); ) {
        InFlightBlock& inflight = g_in_flight_blocks[i];
        cudaError_t err = cudaEventQuery(inflight.event);

        if (err == cudaSuccess) {
            // Event has completed, so the block is safe to reuse.
            g_free_pool[inflight.block.size].push_back(inflight.block);
            cudaEventDestroy(inflight.event);
            
            // Swap with the last element and pop back to remove efficiently
            std::swap(g_in_flight_blocks[i], g_in_flight_blocks.back());
            g_in_flight_blocks.pop_back();
        } else if (err == cudaErrorNotReady) {
            // Event has not completed, check the next one.
            i++;
        } else {
            // An actual error occurred. For now, just print it.
            CkPrintf("WARNING: cudaEventQuery returned error %s for ptr %p\n",
                     cudaGetErrorString(err), inflight.block.ptr);
            i++;
        }
    }
}

/**
 * @brief Initializes the memory pool.
 */
void poolInit() {
    if (g_poolLock == nullptr) {
        g_poolLock = CmiCreateLock();
    }
}

/**
 * @brief Destroys the memory pool and frees all cached memory.
 */
void poolDestroy() {
    if (g_poolLock == nullptr) return;

    CmiLock(g_poolLock);
    
    // Synchronize and free any remaining in-flight blocks
    for (auto& inflight : g_in_flight_blocks) {
        cudaEventSynchronize(inflight.event);
        cudaEventDestroy(inflight.event);
        cudaFree(inflight.block.ptr);
    }
    g_in_flight_blocks.clear();

    // Free all blocks in the free pool
    for (auto const& pair : g_free_pool) {
        for (const auto& block : pair.second) {
            cudaFree(block.ptr);
        }
    }
    g_free_pool.clear();
    g_pointer_sizes.clear();

    CmiUnlock(g_poolLock);
    // Note: The lock itself is not destroyed, as Charm++ does not have a destroy lock primitive.
}

/**
 * @brief Allocates a block of memory from the pool.
 */
cudaError_t poolMalloc(void** devPtr, size_t size, cudaStream_t stream) {
    if (g_poolLock == nullptr) {
        poolInit(); // Ensure pool is initialized
    }

    // Get the bucket size for this allocation
    size_t bucketSize = getBucketSize(size);

    CmiLock(g_poolLock);
    reclaim_completed_blocks();

    auto it = g_free_pool.find(bucketSize);
    if (it != g_free_pool.end() && !it->second.empty()) {
        // Found a suitable block from the bucket, reuse it
        PoolBlock block = it->second.back();
        it->second.pop_back();
        *devPtr = block.ptr;

        g_pointer_sizes[*devPtr] = block.size; // Track actual allocated size for poolFree

        CmiUnlock(g_poolLock);

        // Logging - log with requested size, not bucket size
        DataManager* dm = (DataManager*)CkLocalNodeBranch(dataManagerID);
        if (dm && dm->memLog && dm->bGpuMemLogger) {
            MemLogEvent event(CkMyNode(), MEMLOG_POOL_REUSE, size, (uintptr_t)*devPtr, CkWallTimer(), __FILE__, __LINE__, "pooled_ptr", "poolMalloc");
            CmiLock(dm->lockMemLog);
            dm->memLog->meTab.push_back(event);
            CmiUnlock(dm->lockMemLog);
        }
        
        return cudaSuccess;
    }

    // No suitable block found, allocate a new one at bucket size
    CmiUnlock(g_poolLock);
    
    cudaError_t result = cudaMalloc(devPtr, bucketSize);

    if (result == cudaSuccess) {
        CmiLock(g_poolLock);
        g_pointer_sizes[*devPtr] = bucketSize; // Track actual allocated size (bucket size)
        CmiUnlock(g_poolLock);
    }
    
    // Log the new allocation with requested size
    if (result == cudaSuccess) {
        DataManager* dm = (DataManager*)CkLocalNodeBranch(dataManagerID);
        if (dm && dm->memLog && dm->bGpuMemLogger) {
            MemLogEvent event(CkMyNode(), MEMLOG_ALLOC, size, (uintptr_t)*devPtr, CkWallTimer(), __FILE__, __LINE__, "pooled_ptr", "poolMalloc_fallback");
            CmiLock(dm->lockMemLog);
            dm->memLog->meTab.push_back(event);
            CmiUnlock(dm->lockMemLog);
        }
    }

    return result;
}

/**
 * @brief Returns a block of memory to the pool.
 */
cudaError_t poolFree(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }
    if (g_poolLock == nullptr) {
        // This case should ideally not happen if poolInit is called correctly.
        CkAbort("poolFree called before poolInit or on a non-pooled pointer.");
    }

    CmiLock(g_poolLock);
    auto it = g_pointer_sizes.find(ptr);
    if (it == g_pointer_sizes.end()) {
        CmiUnlock(g_poolLock);
        CkPrintf("WARNING: Attempted to poolFree a pointer not allocated by poolMalloc: %p\n", ptr);
        // Fallback to a direct free, though this indicates a logic error.
        return gpuFreeTracked(ptr, "untracked_ptr", "poolFree_fallback", __FILE__, __LINE__);
    }
    size_t size = it->second;
    g_pointer_sizes.erase(it);
    CmiUnlock(g_poolLock);

    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, stream);

    InFlightBlock inflight;
    inflight.block.ptr = ptr;
    inflight.block.size = size;
    inflight.event = event;

    CmiLock(g_poolLock);
    g_in_flight_blocks.push_back(inflight);
    CmiUnlock(g_poolLock);

    // Logging
    DataManager* dm = (DataManager*)CkLocalNodeBranch(dataManagerID);
    if (dm && dm->memLog && dm->bGpuMemLogger) {
        MemLogEvent log_event(CkMyNode(), MEMLOG_POOL_RELEASE, size, (uintptr_t)ptr, CkWallTimer(), __FILE__, __LINE__, "pooled_ptr", "poolFree");
        CmiLock(dm->lockMemLog);
        dm->memLog->meTab.push_back(log_event);
        CmiUnlock(dm->lockMemLog);
    }

    return cudaSuccess;
}

#endif

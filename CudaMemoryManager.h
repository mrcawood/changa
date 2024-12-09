#ifndef _CUDA_MEMORY_MANAGER_H_
#define _CUDA_MEMORY_MANAGER_H_

#include <cuda_runtime.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>
#include <algorithm>
#include <chrono>
#include "charm++.h"

// Debug print control moved to common header
#include "cuda_debug.h"

#include <fstream>
#include <sstream>

/**
 * @brief Safe memory management for CUDA allocations
 * Handles allocation, tracking, and cleanup of CUDA memory
 */
class CudaMemoryManager {
public:
    // Performance monitoring stats
    struct PoolStats {
        // Per-timestep statistics
        size_t allocationsThisStep{0};     // Allocations in current timestep
        size_t reuseCountThisStep{0};      // Reuses in current timestep
        size_t newAllocationsThisStep{0};  // New allocations in current timestep
        size_t bytesAllocatedThisStep{0};  // Bytes allocated in current timestep
        double avgTimeThisStep{0};         // Average allocation time this step
        size_t peakMemoryThisStep{0};      // Peak memory usage this step
        
        // Lifetime statistics
        size_t totalAllocations{0};        // Total number of allocation requests
        size_t totalReuse{0};              // Total number of buffer reuses
        size_t totalNewAllocations{0};     // Total number of new allocations
        size_t totalBytesAllocated{0};     // Total bytes allocated
        size_t peakMemoryUsage{0};         // Peak memory usage
        double avgAllocationTime{0};        // Average time for allocation in ms
        
        void reset() {
            // Only reset per-timestep stats
            allocationsThisStep = 0;
            reuseCountThisStep = 0;
            newAllocationsThisStep = 0;
            bytesAllocatedThisStep = 0;
            avgTimeThisStep = 0;
            peakMemoryThisStep = 0;
        }
        
        void printTimestepStats(int peId) const {
            // Always print stats, even if no activity
            CUDA_MEM_STATS("\n=== CudaMemoryManager Timestep Statistics (PE %d) ===\n", peId);
            CUDA_MEM_STATS("Allocations this step: %zu\n", allocationsThisStep);
            CUDA_MEM_STATS("Buffer reuse rate: %zu (%.1f%%)\n", 
                     reuseCountThisStep, 
                     allocationsThisStep > 0 ? (reuseCountThisStep * 100.0 / allocationsThisStep) : 0.0);
            
            // Show memory in appropriate units
            if (bytesAllocatedThisStep < 1024*1024) {
                CUDA_MEM_STATS("Memory allocated this step: %.2f KB\n", bytesAllocatedThisStep / 1024.0);
                CUDA_MEM_STATS("Peak memory this step: %.2f KB\n", peakMemoryThisStep / 1024.0);
            } else {
                CUDA_MEM_STATS("Memory allocated this step: %.2f MB\n", bytesAllocatedThisStep / (1024.0 * 1024.0));
                CUDA_MEM_STATS("Peak memory this step: %.2f MB\n", peakMemoryThisStep / (1024.0 * 1024.0));
            }
            
            CUDA_MEM_STATS("Average allocation time: %.3f ms\n", avgTimeThisStep);
            CUDA_MEM_STATS("=====================================\n\n");
        }
        
        void printLifetimeStats(int peId) const {
            CUDA_MEM_STATS("\n=== CudaMemoryManager Lifetime Statistics (PE %d) ===\n", peId);
            CUDA_MEM_STATS("Total allocations: %zu\n", totalAllocations);
            CUDA_MEM_STATS("Total buffer reuses: %zu (%.1f%%)\n", 
                     totalReuse,
                     totalAllocations > 0 ? (totalReuse * 100.0 / totalAllocations) : 0);
            
            // Show memory in appropriate units
            if (totalBytesAllocated < 1024*1024) {
                CUDA_MEM_STATS("Total memory allocated: %.2f KB\n", totalBytesAllocated / 1024.0);
                CUDA_MEM_STATS("Lifetime peak memory: %.2f KB\n", peakMemoryUsage / 1024.0);
            } else {
                CUDA_MEM_STATS("Total memory allocated: %.2f MB\n", totalBytesAllocated / (1024.0 * 1024.0));
                CUDA_MEM_STATS("Lifetime peak memory: %.2f MB\n", peakMemoryUsage / (1024.0 * 1024.0));
            }
            
            CUDA_MEM_STATS("Lifetime average allocation time: %.3f ms\n", avgAllocationTime);
            CUDA_MEM_STATS("=====================================\n\n");
        }
    };

    struct MemoryBlock {
        void* ptr;           // Device pointer
        size_t size;         // Size in bytes
        cudaStream_t stream; // Associated stream
        bool inUse;         // Whether block is currently in use
        std::chrono::steady_clock::time_point lastUsed; // Timestamp of last use
    };

    // Buffer pool with integrated monitoring
    struct BufferPool {
        static const size_t MAX_BUFFERS = 4;  // d_list, markers, starts, sizes
        std::vector<MemoryBlock> available;
        std::vector<MemoryBlock> inUse;
        PoolStats stats;
        std::mutex poolMutex;
        
        void* acquireBuffer(size_t minSize, cudaStream_t stream) {
            auto start = std::chrono::steady_clock::now();
            
            // Update both timestep and lifetime allocation counts
            stats.allocationsThisStep++;
            stats.totalAllocations++;
            
            // First try to find a buffer of exact size
            auto it = std::find_if(available.begin(), available.end(),
                [minSize](const MemoryBlock& block) { return block.size == minSize; });
                
            // If no exact match, try to find a larger buffer
            if (it == available.end()) {
                it = std::find_if(available.begin(), available.end(),
                    [minSize](const MemoryBlock& block) { return block.size >= minSize; });
            }
                
            void* ptr = nullptr;
            if (it != available.end()) {
                MemoryBlock block = *it;
                available.erase(it);
                block.inUse = true;
                block.lastUsed = std::chrono::steady_clock::now();
                inUse.push_back(block);
                ptr = block.ptr;
                
                // Update reuse stats
                stats.reuseCountThisStep++;
                stats.totalReuse++;
            } else {
                cudaError_t err = cudaMalloc(&ptr, minSize);
                if (err != cudaSuccess) {
                    return nullptr;
                }
                
                // Update allocation stats
                stats.newAllocationsThisStep++;
                stats.totalNewAllocations++;
                stats.bytesAllocatedThisStep += minSize;
                stats.totalBytesAllocated += minSize;
                
                // Update peak memory usage
                size_t currentUsage = 0;
                for (const auto& block : inUse) {
                    currentUsage += block.size;
                }
                currentUsage += minSize;
                stats.peakMemoryThisStep = std::max(stats.peakMemoryThisStep, currentUsage);
                stats.peakMemoryUsage = std::max(stats.peakMemoryUsage, currentUsage);
                
                inUse.push_back({ptr, minSize, stream, true, std::chrono::steady_clock::now()});
            }
            
            // Update timing stats
            auto end = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Update both timestep and lifetime averages
            stats.avgTimeThisStep = (stats.avgTimeThisStep * (stats.allocationsThisStep - 1) + duration) 
                                  / stats.allocationsThisStep;
            stats.avgAllocationTime = (stats.avgAllocationTime * (stats.totalAllocations - 1) + duration) 
                                    / stats.totalAllocations;
            
            return ptr;
        }
        
        void releaseBuffer(void* ptr) {
            auto it = std::find_if(inUse.begin(), inUse.end(),
                [ptr](const MemoryBlock& block) { return block.ptr == ptr; });
                
            if (it != inUse.end()) {
                it->inUse = false;
                it->lastUsed = std::chrono::steady_clock::now();
                
                if (available.size() < MAX_BUFFERS) {
                    available.push_back(*it);
                    CUDA_MEM_DEBUG("Returned buffer of size %zu to pool (total available: %zu)\n", 
                             it->size, available.size());
                } else {
                    auto lru = std::min_element(available.begin(), available.end(),
                        [](const MemoryBlock& a, const MemoryBlock& b) {
                            return a.lastUsed < b.lastUsed;
                        });
                        
                    if (lru->lastUsed < it->lastUsed) {
                        cudaFree(lru->ptr);
                        stats.totalBytesAllocated -= lru->size;
                        *lru = *it;
                        CUDA_MEM_DEBUG("Replaced LRU buffer of size %zu with more recent buffer of size %zu\n",
                                 lru->size, it->size);
                    } else {
                        cudaFree(it->ptr);
                        stats.totalBytesAllocated -= it->size;
                        CUDA_MEM_DEBUG("Freed buffer of size %zu (pool full)\n", it->size);
                    }
                }
                
                inUse.erase(it);
            }
        }
        
        void cleanup() {
            std::lock_guard<std::mutex> lock(poolMutex);
            CUDA_MEM_DEBUG("\nFinal pool cleanup - Freeing %zu available and %zu in-use buffers\n",
                         available.size(), inUse.size());
                         
            for (auto& block : available) {
                if (block.ptr) {
                    cudaFree(block.ptr);
                }
            }
            for (auto& block : inUse) {
                if (block.ptr) {
                    cudaFree(block.ptr);
                }
            }
            available.clear();
            inUse.clear();
        }

        // Add back the getStats method
        const PoolStats& getStats() const { return stats; }
    };

    CudaMemoryManager(int peId) : myPeId(peId) {}
    ~CudaMemoryManager() { cleanup(); }

    // Disable copy constructor and assignment
    CudaMemoryManager(const CudaMemoryManager&) = delete;
    CudaMemoryManager& operator=(const CudaMemoryManager&) = delete;

    /**
     * @brief Allocate device memory with tracking
     * @param size Size in bytes to allocate
     * @param name Identifier for the allocation
     * @param stream CUDA stream to use
     * @return Device pointer or nullptr on failure
     */
    void* allocate(size_t size, const std::string& name, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex);
        
        void* ptr = pool.acquireBuffer(size, stream);
        if (!ptr) {
            CUDA_MEM_DEBUG("Failed to allocate %zu bytes for %s\n", size, name.c_str());
            return nullptr;
        }

        CUDA_MEM_DEBUG("Allocated %s at %p size: %zu\n", name.c_str(), ptr, size);
        blocks[name] = {ptr, size, stream, true, std::chrono::steady_clock::now()};
        writeLog("Allocated memory for " + name);
        return ptr;
    }

    /**
     * @brief Async transfer from host to device
     * @param dst Device destination pointer
     * @param src Host source pointer
     * @param size Size in bytes to transfer
     * @param stream CUDA stream to use
     * @return True if transfer initiated successfully
     */
    bool asyncTransfer(void* dst, const void* src, size_t size, cudaStream_t stream) {
        cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            CmiPrintf("Failed async transfer: %s\n", cudaGetErrorString(err));
            return false;
        }
        CUDA_MEM_DEBUG("Started async transfer of %zu bytes\n", size);
        return true;
    }

    /**
     * @brief Free a specific allocation
     * @param name Identifier of allocation to free
     */
    void free(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex);
        
        auto it = blocks.find(name);
        if (it == blocks.end()) {
            CUDA_MEM_DEBUG("Attempted to free unknown block: %s\n", name.c_str());
            return;
        }

        if (it->second.ptr) {
            CUDA_MEM_DEBUG("Freeing %s at %p\n", name.c_str(), it->second.ptr);
            pool.releaseBuffer(it->second.ptr);
        }
        
        blocks.erase(it);
        writeLog("Freed memory for " + name);
    }

    /**
     * @brief Wait for all operations in a stream to complete
     * @param stream CUDA stream to synchronize
     */
    void syncStream(cudaStream_t stream) {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            CmiPrintf("Stream sync failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     * @brief Clean up all allocations
     */
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex);
        pool.cleanup();
        blocks.clear();
    }

    // Print statistics for the current timestep
    void printTimestepStats() const {
        //CmiPrintf("[DEBUG] printTimestepStats called on PE %d\n", myPeId);
        std::lock_guard<std::mutex> lock(mutex);
        pool.getStats().printTimestepStats(myPeId);
    }

    // Print lifetime statistics
    void printLifetimeStats() const {
        std::lock_guard<std::mutex> lock(mutex);
        pool.getStats().printLifetimeStats(myPeId);
    }

    // Print all statistics (both timestep and lifetime)
    void printStats() const {
        std::lock_guard<std::mutex> lock(mutex);
        pool.getStats().printLifetimeStats(myPeId);
    }

    // Reset timestep statistics (call at start of timestep)
    void resetTimestepStats() {
        std::lock_guard<std::mutex> lock(mutex);
        pool.stats.reset();
    }

    void writeLog(const std::string& message) const {
        std::ostringstream oss;
        oss << "node_" << myPeId << "_log.txt";
        std::ofstream logFile(oss.str(), std::ios::app);
        if (logFile.is_open()) {
            logFile << message << std::endl;
            logFile.close();
        }
    }

private:
    int myPeId;  // Processing Element ID
    mutable std::mutex mutex;
    std::unordered_map<std::string, MemoryBlock> blocks;
    BufferPool pool;
};

#endif // _CUDA_MEMORY_MANAGER_H_ 
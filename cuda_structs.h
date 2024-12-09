#ifndef _CUDA_STRUCTS_H_
#define _CUDA_STRUCTS_H_

// Forward declaration of CudaMemoryManager
class CudaMemoryManager;

/// Device memory pointers used by most functions in HostCUDA
struct CudaDevPtr {
    void *d_list;
    void *d_bucketMarkers;
    void *d_bucketStarts;
    void *d_bucketSizes;
    CudaMemoryManager* memManager;
    
    CudaDevPtr() : d_list(nullptr), d_bucketMarkers(nullptr), 
        d_bucketStarts(nullptr), d_bucketSizes(nullptr),
        memManager(nullptr) {}
};

#endif 
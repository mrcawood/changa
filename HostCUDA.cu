#ifdef _WIN32
#define NOMINMAX
#endif

#ifdef HAPI_MEMPOOL
#define GPU_MEMPOOL
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
// #include <cutil.h>
#include <assert.h>

#include "CudaFunctions.h"
#include "CUDAMoments.cu"
#include "HostCUDA.h"
#include "EwaldCUDA.h"

#include "hapi.h"
#include "cuda_typedef.h"
#include "cuda/intrinsics/voting.hu"
#include "cuda/intrinsics/shfl.hu"
#include "GPUMemoryPool.h"

#ifdef GPU_LOCAL_TREE_WALK
#include "codes.h"
#endif //GPU_LOCAL_TREE_WALK

#ifdef HAPI_TRACE
#  define HAPI_TRACE_BEGIN()   double trace_start_time = CmiWallTimer()
#  define HAPI_TRACE_END(ID)   traceUserBracketEvent(ID, trace_start_time, CmiWallTimer())
#else
#  define HAPI_TRACE_BEGIN() /* */
#  define HAPI_TRACE_END(ID) /* */
#endif

#define cudaChk(code) cudaErrorDie(code, #code, __FILE__, __LINE__)
inline void cudaErrorDie(cudaError_t retCode, const char* code,
                                              const char* file, int line) {
  if (retCode != cudaSuccess) {
    fprintf(stderr, "Fatal CUDA Error %s at %s:%d.\nReturn value %d from '%s'.",
        cudaGetErrorString(retCode), file, line, retCode, code);
    abort();
  }
}

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
#include "converse.h"
#endif

__device__ __constant__ EwaldReadOnlyData cachedData[1];
__device__ __constant__ EwtData ewt[NEWH];  


//__constant__ constantData[88];
//
//
#ifdef HAPI_TRACE
extern "C" void traceUserBracketEvent(int e, double beginT, double endT);
extern "C" double CmiWallTimer();
#endif


void allocatePinnedHostMemory(void **ptr, size_t size){
  if(size <= 0){
    *((char **)ptr) = NULL;
#ifdef CUDA_PRINT_ERRORS
    printf("allocatePinnedHostMemory: 0 size!\n");
#endif
    assert(0);
    return;
  }
#ifdef HAPI_MEMPOOL
  hapiMallocHost(ptr, size, true);
#else
  hapiMallocHost(ptr, size, false);
#endif
#ifdef CUDA_PRINT_ERRORS
  printf("allocatePinnedHostMemory: %s size: %zu\n", cudaGetErrorString( cudaGetLastError() ), size);
#endif
}

void freePinnedHostMemory(void *ptr){
  if(ptr == NULL){
#ifdef CUDA_PRINT_ERRORS
    printf("freePinnedHostMemory: NULL ptr!\n");
#endif
    assert(0);
    return;
  }
#ifdef HAPI_MEMPOOL
  hapiFreeHost(ptr, true);
#else
  hapiFreeHost(ptr, false);
#endif
#ifdef CUDA_PRINT_ERRORS
  printf("freePinnedHostMemory: %s\n", cudaGetErrorString( cudaGetLastError() ));
#endif
}

/// @brief Transfer local moments, particle data and acceleration fields to GPU memory
/// @param moments Array of moments
/// @param sMoments Size of moments array
/// @param compactParts Array of particles
/// @param sCompactParts Size of particle array
/// @param varParts Zeroed-out particle acceleration fields
/// @param sVarParts Size of acceleration array
/// @param d_localMoments Uninitalized pointer to moments on GPU
/// @param d_compactParts Uninitalized pointer to particles on GPU
/// @param d_varParts Uninitalized pointer to accelerations on GPU
/// @param stream CUDA stream to handle the memory transfer
/// @param numParticles Total number of particle accelerations to initalize
void DataManagerTransferLocalTree(void *moments, size_t sMoments,
                                  void *compactParts, size_t sCompactParts,
                                  void *varParts, size_t sVarParts,
				  void **d_localMoments, void **d_compactParts, void **d_varParts,
				  cudaStream_t stream, int numParticles,
                                  void *callback) {

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) DM LOCAL TREE moments %zu partcores %zu partvars %zu\n",
           CmiMyPe(),
           sMoments,
           sCompactParts,
           sVarParts
           );
#endif

  HAPI_TRACE_BEGIN();

  cudaChk(cudaMalloc(d_localMoments, sMoments));
  cudaChk(cudaMalloc(d_compactParts, sCompactParts));
  cudaChk(cudaMalloc(d_varParts, sVarParts));

  cudaChk(cudaMemcpyAsync(*d_localMoments, moments, sMoments, cudaMemcpyHostToDevice, stream));
  cudaChk(cudaMemcpyAsync(*d_compactParts, compactParts, sCompactParts, cudaMemcpyHostToDevice, stream));
  cudaChk(cudaMemcpyAsync(*d_varParts, varParts, sVarParts, cudaMemcpyHostToDevice, stream));

  ZeroVars<<<numParticles / THREADS_PER_BLOCK + 1, dim3(THREADS_PER_BLOCK), 0, stream>>>(
      (VariablePartData *) *d_varParts,
      numParticles);
  cudaChk(cudaPeekAtLastError());

  HAPI_TRACE_END(CUDA_XFER_LOCAL);

  hapiAddCallback(stream, callback);
}

/// @brief Transfer remote moments and particle data to GPU memory
/// @param moments Array of remote moments
/// @param sMoments Size of remote moments array
/// @param remoteParts Array of remote particles
/// @param sRemoteParts Size of remote particle array
/// @param d_remoteMoments Uninitalized pointer to remote moments on GPU
/// @param d_remoteParts Uninitalized pointer to remote particles on GPU
/// @param stream CUDA stream to handle the memory transfer
void DataManagerTransferRemoteChunk(void *moments, size_t sMoments,
                                    void *remoteParts, size_t sRemoteParts,
				    void **d_remoteMoments, void **d_remoteParts,
                                    cudaStream_t stream,
                                    void *callback) {

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) DM REMOTE CHUNK moments %zu partcores %zu\n",
        CmiMyPe(),
        sMoments,
        sRemoteParts
        );
#endif

  HAPI_TRACE_BEGIN();

  cudaChk(cudaMalloc(d_remoteMoments, sMoments));
  cudaChk(cudaMalloc(d_remoteParts, sRemoteParts));
  cudaChk(cudaMemcpyAsync(*d_remoteMoments, moments, sMoments, cudaMemcpyHostToDevice, stream));
  cudaChk(cudaMemcpyAsync(*d_remoteParts, remoteParts, sRemoteParts, cudaMemcpyHostToDevice, stream));

  HAPI_TRACE_END(CUDA_XFER_REMOTE);

  hapiAddCallback(stream, callback);
}

/************** Gravity *****************/

/// @brief Initiate a local gravity calculation on the GPU, via an interaction
///        list calculation between nodes, or do a local tree walk
/// @param data CudaRequest object containing parameters for the calculation
void TreePieceCellListDataTransferLocal(CudaRequest *data){
  cudaStream_t stream = data->stream;
  CudaDevPtr devPtr;
  TreePieceDataTransferBasic(data, &devPtr);

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) TRANSFER LOCAL CELL\n", CmiMyPe());
#endif

#ifdef CUDA_NOTIFY_DATA_TRANSFER_DONE
  printf("TRANSFER LOCAL CELL KERNELSELECT buffers:\nlocal_particles: (0x%x)\nlocal_particle_vars: (0x%x)\nremote_moments: (0x%x)\nil_cell: (0x%x)\n", 
	data->d_localParts,
	data->d_localVars,
	data->d_localMoments,
	devPtr.d_list
      );
#endif

  HAPI_TRACE_BEGIN();
#ifndef CUDA_NO_KERNELS
#ifdef GPU_LOCAL_TREE_WALK
  gpuLocalTreeWalk<<<(data->lastParticle - data->firstParticle + 1)
                     / THREADS_PER_BLOCK + 1, dim3(THREADS_PER_BLOCK), 0, stream>>> (
    data->d_localMoments,
    data->d_localParts,
    data->d_localVars,
    data->firstParticle,
    data->lastParticle,
    data->rootIdx,
    data->theta,
    data->thetaMono,
    data->nReplicas,
    data->fperiod,
    data->fperiodY,
    data->fperiodZ
    );
 #else
    dim3 dimensions = THREADS_PER_BLOCK;
 #ifdef CUDA_2D_TB_KERNEL
    dimensions = dim3(NODES_PER_BLOCK, PARTS_PER_BLOCK);
 #endif
  nodeGravityComputation<<<dim3(data->numBucketsPlusOne-1), dimensions, 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    data->d_localMoments,
    (ILCell *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#endif
#endif
  TreePieceDataTransferBasicCleanup(&devPtr);
  cudaChk(cudaPeekAtLastError());
  HAPI_TRACE_END(CUDA_GRAV_LOCAL);

  hapiAddCallback(stream, data->cb);
}

/// @brief Initiate a remote gravity calculation on the GPU between tree nodes
/// @param data CudaRequest object containing parameters for the calculation
void TreePieceCellListDataTransferRemote(CudaRequest *data){
  cudaStream_t stream = data->stream;
  CudaDevPtr devPtr;
  TreePieceDataTransferBasic(data, &devPtr);

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) TRANSFER REMOTE CELL\n", CmiMyPe());
#endif

#ifdef CUDA_NOTIFY_DATA_TRANSFER_DONE
  printf("TRANSFER REMOTE CELL KERNELSELECT buffers:\nlocal_particles: (0x%x)\nlocal_particle_vars: (0x%x)\nremote_moments: (0x%x)\nil_cell: (0x%x)\n", 
	data->d_localParts,
	data->d_localVars,
	data->d_remoteMoments,
	devPtr.d_list
        );
#endif

  HAPI_TRACE_BEGIN();
#ifndef CUDA_NO_KERNELS
  dim3 dimensions = THREADS_PER_BLOCK;
#ifdef CUDA_2D_TB_KERNEL
  dimensions = dim3(NODES_PER_BLOCK, PARTS_PER_BLOCK);
#endif
  nodeGravityComputation<<<data->numBucketsPlusOne-1, dimensions, 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    data->d_remoteMoments,
    (ILCell *)devPtr.d_list, 
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    ); 
#endif
  TreePieceDataTransferBasicCleanup(&devPtr);
  cudaChk(cudaPeekAtLastError());
  HAPI_TRACE_END(CUDA_GRAV_REMOTE);

  hapiAddCallback(stream, data->cb);
}

/// @brief Initiate a remote resume gravity calculation on the GPU between tree nodes
/// @param data CudaRequest object containing parameters for the calculation
void TreePieceCellListDataTransferRemoteResume(CudaRequest *data){
  cudaStream_t stream = data->stream;
  CudaDevPtr devPtr;
  void *d_missedNodes;
  TreePieceDataTransferBasic(data, &devPtr);

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) TRANSFER REMOTE RESUME CELL\n", CmiMyPe());
#endif

  cudaChk(cudaMalloc(&d_missedNodes, data->sMissed));
  cudaChk(cudaMemcpyAsync(d_missedNodes, data->missedNodes, data->sMissed, cudaMemcpyHostToDevice, stream));

#ifdef CUDA_NOTIFY_DATA_TRANSFER_DONE
  printf("TRANSFER REMOTE RESUME CELL KERNELSELECT buffers:\nlocal_particles: (0x%x)\nlocal_particle_vars: (0x%x)\nmissed_moments (0x%x)\nil_cell (0x%x)\n", 
        data->d_localParts,
	data->d_localVars,
	d_missedNodes,
	devPtr.d_list
      );
#endif

  HAPI_TRACE_BEGIN();
#ifndef CUDA_NO_KERNELS
    dim3 dimensions = THREADS_PER_BLOCK;
#ifdef CUDA_2D_TB_KERNEL
    dimensions = dim3(NODES_PER_BLOCK, PARTS_PER_BLOCK);
#endif
    nodeGravityComputation<<<data->numBucketsPlusOne-1, dimensions, 0, stream>>> (
      data->d_localParts,
      data->d_localVars,
      (CudaMultipoleMoments *)d_missedNodes,
      (ILCell *)devPtr.d_list,
      devPtr.d_bucketMarkers,
      devPtr.d_bucketStarts,
      devPtr.d_bucketSizes,
      data->fperiod
      );
#endif
  TreePieceDataTransferBasicCleanup(&devPtr);
  cudaChk(cudaFree(d_missedNodes));
  cudaChk(cudaPeekAtLastError());
  HAPI_TRACE_END(CUDA_REMOTE_RESUME);

  hapiAddCallback(stream, data->cb);
}

/// @brief Initiate a small phase local gravity calculation on the GPU between particles
/// @param data CudaRequest object containing parameters for the calculation
void TreePiecePartListDataTransferLocalSmallPhase(CudaRequest *data, CompactPartData *particles, int len){
  cudaStream_t stream = data->stream;
  CudaDevPtr devPtr;
  TreePieceDataTransferBasic(data, &devPtr);

  size_t size = (len) * sizeof(CompactPartData);
  void* bufferHostBuffer;
  void* d_smallParts;

#ifdef CUDA_NOTIFY_DATA_TRANSFER_DONE
  printf("TreePiecePartListDataTransferLocalSmallPhase KERNELSELECT buffers:\nlocal_particles: (0x%x)\nlocal_particle_vars: (0x%x)\nil_cell: (0x%x)\n",
      data->d_localParts,
      data->d_localVars,
      devPtr.d_list
      );
#endif

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) TRANSFER LOCAL SMALL PHASE  %zu\n",
      CmiMyPe(),
      size
      );
#endif

  HAPI_TRACE_BEGIN();
  allocatePinnedHostMemory(&bufferHostBuffer, size);
#ifdef CUDA_PRINT_ERRORS
  printf("TPPartSmallPhase 0: %s\n", cudaGetErrorString( cudaGetLastError() ) );
#endif
  memcpy(bufferHostBuffer, particles, size);
  cudaChk(cudaMalloc(&d_smallParts, size));
  cudaChk(cudaMemcpyAsync(d_smallParts, bufferHostBuffer, size, cudaMemcpyHostToDevice, stream));

#ifndef CUDA_NO_KERNELS
#ifdef CUDA_2D_TB_KERNEL
  particleGravityComputation<<<data->numBucketsPlusOne-1, dim3(NODES_PER_BLOCK_PART, PARTS_PER_BLOCK_PART), 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    (CompactPartData *)d_smallParts,
    (ILCell *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#else
  particleGravityComputation<<<data->numBucketsPlusOne-1, THREADS_PER_BLOCK, 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    (CompactPartData *)d_smallParts,
    (ILCell *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#endif
#endif
  TreePieceDataTransferBasicCleanup(&devPtr);
  cudaChk(cudaPeekAtLastError());
  HAPI_TRACE_END(CUDA_PART_GRAV_LOCAL_SMALL);
  cudaChk(cudaFree(d_smallParts));
  hapiAddCallback(stream, data->cb);
}

/// @brief Initiate a local gravity calculation on the GPU between particles
/// @param data CudaRequest object containing parameters for the calculation
void TreePiecePartListDataTransferLocal(CudaRequest *data){
  cudaStream_t stream = data->stream;
  CudaDevPtr devPtr;
  TreePieceDataTransferBasic(data, &devPtr);

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) TRANSFER LOCAL LARGEPHASE PART\n", CmiMyPe());
#endif

#ifdef CUDA_NOTIFY_DATA_TRANSFER_DONE
  printf("TreePiecePartListDataTransferLocal buffers:\nlocal_particles: (0x%x)\nlocal_particle_vars: (0x%x)\nil_cell: (0x%x)\n",
        data->d_localParts,
        data->d_localVars,
        devPtr.d_list
        );
#endif

  HAPI_TRACE_BEGIN();
#ifndef CUDA_NO_KERNELS
#ifdef CUDA_2D_TB_KERNEL
  particleGravityComputation<<<data->numBucketsPlusOne-1, dim3(NODES_PER_BLOCK_PART, PARTS_PER_BLOCK_PART), 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    data->d_localParts,
    (ILCell *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#else
  particleGravityComputation<<<data->numBucketsPlusOne-1, THREADS_PER_BLOCK, 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    data->d_localParts,
    (ILPart *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#endif
#endif
  TreePieceDataTransferBasicCleanup(&devPtr);
  cudaChk(cudaPeekAtLastError());
  HAPI_TRACE_END(CUDA_PART_GRAV_LOCAL);

  hapiAddCallback(stream, data->cb);
}

/// @brief Initiate a remote gravity calculation on the GPU between particles
/// @param data CudaRequest object containing parameters for the calculation
void TreePiecePartListDataTransferRemote(CudaRequest *data){
  cudaStream_t stream = data->stream;
  CudaDevPtr devPtr;
  TreePieceDataTransferBasic(data, &devPtr);

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) TRANSFER REMOTE PART\n", CmiMyPe());
#endif

#ifdef CUDA_NOTIFY_DATA_TRANSFER_DONE
  printf("TreePiecePartListDataTransferRemote KERNELSELECT buffers:\nlocal_particles: (0x%x)\nlocal_particle_vars: (0x%x)\nil_cell: (0x%x) (0x%x)\n",
        data->d_localParts,
        data->d_localVars,
	data->list,
        devPtr.d_list
        );
#endif

  HAPI_TRACE_BEGIN();
#ifndef CUDA_NO_KERNELS
#ifdef CUDA_2D_TB_KERNEL
  particleGravityComputation<<<data->numBucketsPlusOne-1, dim3(NODES_PER_BLOCK_PART, PARTS_PER_BLOCK_PART), 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    data->d_remoteParts,
    (ILCell *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#else
  particleGravityComputation<<<data->numBucketsPlusOne-1, THREADS_PER_BLOCK, 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    data->d_remoteParts,
    (ILPart *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#endif
#endif
  TreePieceDataTransferBasicCleanup(&devPtr);
  cudaChk(cudaPeekAtLastError());
  HAPI_TRACE_END(CUDA_PART_GRAV_REMOTE);

  hapiAddCallback(stream, data->cb);
}

/// @brief Initiate a remote gravity calculation on the GPU between particles
/// @param data CudaRequest object containing parameters for the calculation
void TreePiecePartListDataTransferRemoteResume(CudaRequest *data){
  cudaStream_t stream = data->stream;
  CudaDevPtr devPtr;
  void* d_missedParts;
  TreePieceDataTransferBasic(data, &devPtr);

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("(%d) TRANSFER REMOTE RESUME PART\n", CmiMyPe());
#endif

  cudaChk(cudaMalloc(&d_missedParts, data->sMissed));
  cudaChk(cudaMemcpyAsync(d_missedParts, data->missedParts, data->sMissed, cudaMemcpyHostToDevice, stream));

#ifdef CUDA_NOTIFY_DATA_TRANSFER_DONE
  printf("TreePiecePartListDataTransferRemoteResume KERNELSELECT buffers:\nlocal_particles: (0x%x)\nlocal_particle_vars: (0x%x)\nmissed_parts (0x%x)\nil_cell: (0x%x) (0x%x)\n", 
        data->d_localParts,
        data->d_localVars,
        (CompactPartData *)d_missedParts,
	data->list,
        devPtr.d_list
        );
#endif

  HAPI_TRACE_BEGIN();
#ifndef CUDA_NO_KERNELS
#ifdef CUDA_2D_TB_KERNEL
  particleGravityComputation<<<data->numBucketsPlusOne-1, dim3(NODES_PER_BLOCK_PART, PARTS_PER_BLOCK_PART), 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    (CompactPartData *)d_missedParts,
    (ILCell *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#else
  particleGravityComputation<<<data->numBucketsPlusOne-1, THREADS_PER_BLOCK, 0, stream>>> (
    data->d_localParts,
    data->d_localVars,
    (CompactPartData *)d_missedParts,
    (ILPart *)devPtr.d_list,
    devPtr.d_bucketMarkers,
    devPtr.d_bucketStarts,
    devPtr.d_bucketSizes,
    data->fperiod
    );
#endif
#endif
  TreePieceDataTransferBasicCleanup(&devPtr);
  cudaChk(cudaFree(d_missedParts));
  cudaChk(cudaPeekAtLastError());
  HAPI_TRACE_END(CUDA_PART_GRAV_REMOTE);

  hapiAddCallback(stream, data->cb);
}

/// @brief Allocate space and copy bucket and interaction list data to
///         device memory
/// @param data CudaRequest object containing parameters for the calculation
/// @param ptr CudaDevPtr object that stores handles to device memory
void TreePieceDataTransferBasic(CudaRequest *data, CudaDevPtr *ptr){
  cudaStream_t stream = data->stream;

  int numBucketsPlusOne = data->numBucketsPlusOne;
  int numBuckets = numBucketsPlusOne-1;
  size_t listSize = (data->numInteractions) * sizeof(ILCell);
  size_t markerSize = (numBucketsPlusOne) * sizeof(int);
  size_t startSize = (numBuckets) * sizeof(int);

  //cudaChk(cudaMalloc(&ptr->d_list, listSize));
  //cudaChk(cudaMalloc(&ptr->d_bucketMarkers, markerSize));
  //cudaChk(cudaMalloc(&ptr->d_bucketStarts, startSize));
  //cudaChk(cudaMalloc(&ptr->d_bucketSizes, startSize));

  gpuPoolMalloc(&ptr->d_list, listSize);
  gpuPoolMalloc(reinterpret_cast<void**>(&ptr->d_bucketMarkers), markerSize);
  gpuPoolMalloc(reinterpret_cast<void**>(&ptr->d_bucketStarts), startSize);
  gpuPoolMalloc(reinterpret_cast<void**>(&ptr->d_bucketSizes), startSize);


  cudaChk(cudaMemcpyAsync(ptr->d_list, data->list, listSize, cudaMemcpyHostToDevice, stream));
  cudaChk(cudaMemcpyAsync(ptr->d_bucketMarkers, data->bucketMarkers, markerSize, cudaMemcpyHostToDevice, stream));
  cudaChk(cudaMemcpyAsync(ptr->d_bucketStarts, data->bucketStarts, startSize, cudaMemcpyHostToDevice, stream));
  cudaChk(cudaMemcpyAsync(ptr->d_bucketSizes, data->bucketSizes, startSize, cudaMemcpyHostToDevice, stream));

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
    printf("(%d) TRANSFER BASIC %zu bucket_markers %zu bucket_starts %zu\n",
           CmiMyPe(),
           listSize,
           markerSize,
           startSize
           );
#endif
}

/// @brief Free device memory used for interaction list and bucket data
/// @param ptr CudaDevPtr object that stores handles to device memory
void TreePieceDataTransferBasicCleanup(CudaDevPtr *ptr){
  //cudaChk(cudaFree(ptr->d_list));
  //cudaChk(cudaFree(ptr->d_bucketMarkers));
  //cudaChk(cudaFree(ptr->d_bucketStarts));
  //cudaChk(cudaFree(ptr->d_bucketSizes));
  gpuPoolFree(ptr->d_list);
  gpuPoolFree(ptr->d_bucketMarkers);
  gpuPoolFree(ptr->d_bucketStarts);
  gpuPoolFree(ptr->d_bucketSizes);
}

/** @brief Transfer forces from the GPU back to the host. Also schedules
 *         the freeing of the device buffers used for the force calculation.
 *  @param hostBuffer Buffer to store results.
 *  @param size hostBuffer size.
 *  @param d_varParts Pointer to finalized accelerations on GPU
 *  @param stream CUDA stream to handle the memory transfer
 *  @param cb Callback when transfer is done.
 */
void TransferParticleVarsBack(VariablePartData *hostBuffer, size_t size, void *d_varParts,
                              cudaStream_t stream, void *cb){
  
  HAPI_TRACE_BEGIN();
  cudaChk(cudaMemcpyAsync(hostBuffer, d_varParts, size, cudaMemcpyDeviceToHost, stream));
  HAPI_TRACE_END(CUDA_XFER_BACK);
  hapiAddCallback(stream, cb);
}

/*
void DummyKernel(void *cb){
  hapiWorkRequest* dummy = hapiCreateWorkRequest();

  dummy->setExecParams(1, THREADS_PER_BLOCK);

  dummy->setDeviceToHostCallback(cb);
#ifdef HAPI_TRACE
  dummy->setTraceName("dummyRun");
#endif
  dummy->setRunKernel(run_kernel_DUMMY);
  hapiEnqueue(dummy);

}
*/

/*
 * Kernels
 */

/****
 * GPU Local tree walk (computation integrated)
****/
#ifdef GPU_LOCAL_TREE_WALK
__device__ __forceinline__ void
ldgTreeNode(CUDATreeNode &m, CudaMultipoleMoments *ptr) {
  m.radius        = __ldg(&(ptr->radius));
  m.soft          = __ldg(&(ptr->soft));
  m.totalMass     = __ldg(&(ptr->totalMass));
  m.cm.x          = __ldg(&(ptr->cm.x));
  m.cm.y          = __ldg(&(ptr->cm.y));
  m.cm.z          = __ldg(&(ptr->cm.z));
  m.bucketStart   = __ldg(&(ptr->bucketStart));
  m.bucketSize    = __ldg(&(ptr->bucketSize));
  m.particleCount = __ldg(&(ptr->particleCount));
  m.children[0]   = __ldg(&(ptr->children[0]));
  m.children[1]   = __ldg(&(ptr->children[1]));
  m.type          = __ldg(&(ptr->type));
};

__device__ __forceinline__ void
ldgBucketNode(CUDABucketNode &m, CompactPartData *ptr) {
  m.soft      = __ldg(&(ptr->soft));
  m.totalMass = __ldg(&(ptr->mass));
  m.cm.x      = __ldg(&(ptr->position.x));
  m.cm.y      = __ldg(&(ptr->position.y));
  m.cm.z      = __ldg(&(ptr->position.z));
};

__device__ __forceinline__ void
ldgBucketNode(CUDABucketNode &m, CudaMultipoleMoments *ptr) {
  m.radius    = __ldg(&(ptr->radius));
  m.soft      = __ldg(&(ptr->soft));
  m.totalMass = __ldg(&(ptr->totalMass));
  m.cm.x      = __ldg(&(ptr->cm.x));
  m.cm.y      = __ldg(&(ptr->cm.y));
  m.cm.z      = __ldg(&(ptr->cm.z));

  m.lesser_corner.x   = __ldg(&(ptr->lesser_corner.x));
  m.lesser_corner.y   = __ldg(&(ptr->lesser_corner.y));
  m.lesser_corner.z   = __ldg(&(ptr->lesser_corner.z));

  m.greater_corner.x  = __ldg(&(ptr->greater_corner.x));
  m.greater_corner.y  = __ldg(&(ptr->greater_corner.y));
  m.greater_corner.z  = __ldg(&(ptr->greater_corner.z));
};

__device__ __forceinline__ void
ldgParticle(CompactPartData &m, CompactPartData *ptr) {
  m.mass       = __ldg(&(ptr->mass));
  m.soft       = __ldg(&(ptr->soft));
  m.position.x = __ldg(&(ptr->position.x));
  m.position.y = __ldg(&(ptr->position.y));
  m.position.z = __ldg(&(ptr->position.z));
}

__device__ __forceinline__ void stackInit(int &sp, int* stk, int rootIdx) {
  sp = 0;
  stk[sp] = rootIdx;
}

__device__ __forceinline__ void stackPush(int &sp) {
  ++sp;
}

__device__ __forceinline__ void stackPop(int &sp) {
  --sp;
}

const int stackDepth = 64;

//__launch_bounds__(1024,1)
__global__ void gpuLocalTreeWalk(
  CudaMultipoleMoments *moments,
  CompactPartData *particleCores,
  VariablePartData *particleVars,
  int firstParticle,
  int lastParticle,
  int rootIdx,
  cudatype theta,
  cudatype thetaMono,
  int nReplicas,
  cudatype fperiod,
  cudatype fperiodY,
  cudatype fperiodZ) {

  CUDABucketNode  myNode;
  CompactPartData myParticle;

#if __CUDA_ARCH__ >= 700
  // Non-lockstepping code for Volta GPUs
  int sp;
  int stk[stackDepth];
  CUDATreeNode targetNode;

#define SP sp
#define STACK_TOP_INDEX stk[SP]
#define TARGET_NODE targetNode
#else
  // Default lockstepping code
  __shared__ int sp[WARPS_PER_BLOCK];
  __shared__ int stk[WARPS_PER_BLOCK][stackDepth];
  __shared__ CUDATreeNode targetNode[WARPS_PER_BLOCK];

#define SP sp[WARP_INDEX]
#define STACK_TOP_INDEX stk[WARP_INDEX][SP]
#define TARGET_NODE targetNode[WARP_INDEX]
#endif

  CudaVector3D acc = {0,0,0};
  cudatype pot = 0;
  cudatype idt2 = 0;
  CudaVector3D offset = {0,0,0};

  int targetIndex = -1;

  // variables for CUDA_momEvalFmomrcm
  CudaVector3D r;
  cudatype rsq = 0;
  cudatype twoh = 0;

  int flag = 1;
  int critical = stackDepth;
  int cond = 1;

  for(int pidx = blockIdx.x*blockDim.x + threadIdx.x + firstParticle;
      pidx <= lastParticle; pidx += gridDim.x*blockDim.x) {
    
    // initialize the variables belonging to current thread
    int nodePointer = particleCores[pidx].nodeId;
    ldgParticle(myParticle, &particleCores[pidx]);
    ldgBucketNode(myNode, &moments[nodePointer]);

    for(int x = -nReplicas; x <= nReplicas; x++) {
      for(int y = -nReplicas; y <= nReplicas; y++) {
        for(int z = -nReplicas; z <= nReplicas; z++) {
          // generate the offset for the periodic boundary conditions 
          offset.x = x*fperiod;
          offset.y = y*fperiodY;
          offset.z = z*fperiodZ;

          flag = 1;
          critical = stackDepth;
          cond = 1;
#if __CUDA_ARCH__ >= 700
          stackInit(SP, stk, rootIdx);
#else
          stackInit(SP, stk[WARP_INDEX], rootIdx);
#endif
          while(SP >= 0) {
            if (flag == 0 && critical >= SP) {
              flag = 1;
            }

            targetIndex = STACK_TOP_INDEX;
            stackPop(SP);

            if (flag) {
              ldgTreeNode(TARGET_NODE, &moments[targetIndex]);
              // Each warp increases its own TARGET_NODE in the shared memory,
              // so there is no actual data racing here.
              addCudaVector3D(TARGET_NODE.cm, offset, TARGET_NODE.cm);

              int open = CUDA_openCriterionNode(TARGET_NODE, myNode, -1, theta,
                                                thetaMono);
              int action = CUDA_OptAction(open, TARGET_NODE.type);

              critical = SP;
              cond = ((action == KEEP) && (open == CONTAIN || open == INTERSECT));

              if (action == COMPUTE) {
                if (CUDA_openSoftening(TARGET_NODE, myNode)) {
                  r.x = TARGET_NODE.cm.x - myParticle.position.x;
                  r.y = TARGET_NODE.cm.y - myParticle.position.y;
                  r.z = TARGET_NODE.cm.z - myParticle.position.z;

                  rsq = r.x*r.x + r.y*r.y + r.z*r.z;
                  twoh = TARGET_NODE.soft + myParticle.soft;
                  cudatype a, b;
                  if (rsq != 0) {
                    CUDA_SPLINE(rsq, twoh, a, b);
                    idt2 = fmax(idt2, (myParticle.mass + TARGET_NODE.totalMass) * b);

                    pot -= TARGET_NODE.totalMass * a;

                    acc.x += r.x*b*TARGET_NODE.totalMass;
                    acc.y += r.y*b*TARGET_NODE.totalMass;
                    acc.z += r.z*b*TARGET_NODE.totalMass;
                  }
                } else {
                  // compute with the node targetnode
                  r.x = myParticle.position.x - TARGET_NODE.cm.x;
                  r.y = myParticle.position.y - TARGET_NODE.cm.y;
                  r.z = myParticle.position.z - TARGET_NODE.cm.z;

                  rsq = r.x*r.x + r.y*r.y + r.z*r.z;
                  if (rsq != 0) {
                    cudatype dir = rsqrt(rsq);
#if defined (HEXADECAPOLE)
                    CUDA_momEvalFmomrcm(&moments[targetIndex], &r, dir, &acc, &pot);
                    idt2 = fmax(idt2, (myParticle.mass +
                                       moments[targetIndex].totalMass)*dir*dir*dir);
#else
                    cudatype a, b, c, d;
                    twoh = moments[targetIndex].soft + myParticle.soft;
                    CUDA_SPLINEQ(dir, rsq, twoh, a, b, c, d);

                    cudatype qirx = moments[targetIndex].xx*r.x +
                                    moments[targetIndex].xy*r.y +
                                    moments[targetIndex].xz*r.z;
                    cudatype qiry = moments[targetIndex].xy*r.x +
                                    moments[targetIndex].yy*r.y +
                                    moments[targetIndex].yz*r.z;
                    cudatype qirz = moments[targetIndex].xz*r.x +
                                    moments[targetIndex].yz*r.y +
                                    moments[targetIndex].zz*r.z;
                    cudatype qir = 0.5 * (qirx*r.x + qiry*r.y + qirz*r.z);
                    cudatype tr = 0.5 * (moments[targetIndex].xx +
                                         moments[targetIndex].yy +
                                         moments[targetIndex].zz);
                    cudatype qir3 = b*moments[targetIndex].totalMass + d*qir - c*tr;

                    pot -= moments[targetIndex].totalMass * a + c*qir - b*tr;
                    acc.x -= qir3*r.x - c*qirx;
                    acc.y -= qir3*r.y - c*qiry;
                    acc.z -= qir3*r.z - c*qirz;
                    idt2 = fmax(idt2, (myParticle.mass +
                                       moments[targetIndex].totalMass) * b);
#endif //HEXADECAPOLE
                  }
                }
              } else if (action == KEEP_LOCAL_BUCKET) {
                // compute with each particle contained by node targetnode
                int target_firstparticle = TARGET_NODE.bucketStart;
                int target_lastparticle = TARGET_NODE.bucketStart +
                                          TARGET_NODE.bucketSize;
                cudatype a, b;
                for (int i = target_firstparticle; i < target_lastparticle; ++i) {
                  CompactPartData targetParticle = particleCores[i];
                  addCudaVector3D(targetParticle.position, offset, targetParticle.position);

                  r.x = targetParticle.position.x - myParticle.position.x;
                  r.y = targetParticle.position.y - myParticle.position.y;
                  r.z = targetParticle.position.z - myParticle.position.z;

                  rsq = r.x*r.x + r.y*r.y + r.z*r.z;
                  twoh = targetParticle.soft + myParticle.soft;
                  if (rsq != 0) {
                    CUDA_SPLINE(rsq, twoh, a, b);
                    pot -= targetParticle.mass * a;

                    acc.x += r.x*b*targetParticle.mass;
                    acc.y += r.y*b*targetParticle.mass;
                    acc.z += r.z*b*targetParticle.mass;
                    idt2 = fmax(idt2, (myParticle.mass + targetParticle.mass) * b);
                  }
                }
              }

              if (!any(cond)) {
                continue;
              }

              if (!cond) {
                flag = 0;
              } else {
                if (TARGET_NODE.children[1] != -1) {
                  stackPush(SP);
                  STACK_TOP_INDEX = TARGET_NODE.children[1];
                }
                if (TARGET_NODE.children[0] != -1) {
                  stackPush(SP);
                  STACK_TOP_INDEX = TARGET_NODE.children[0];
                }
              }
            }
#if __CUDA_ARCH__ >= 700
            __syncwarp();
#endif
          }
        } // z replicas
      } // y replicas
    } // x replicas

    particleVars[pidx].a.x += acc.x;
    particleVars[pidx].a.y += acc.y;
    particleVars[pidx].a.z += acc.z;
    particleVars[pidx].potential += pot;
    particleVars[pidx].dtGrav = fmax(idt2,  particleVars[pidx].dtGrav);
  }
}
#endif //GPU_LOCAL_TREE_WALK

/**
 * @brief interaction between multipole moments and buckets of particles.
 * @param particleCores Read-only properties of particles.
 * @param particleVars Accumulators of accelerations etc. of particles.
 * @param moments Multipole moments from which to calculate forces.
 * @param ils Cells on the interaction list.  Each Cell has an index into
 *            moments.
 * @param ilmarks Indices into ils for each block.
 * @param bucketStarts Indices into particleCores and particleVars
 *                      for each block
 * @param bucketSizes Size of the bucket for each block
 * @param fPeriod Size of periodic boundary condition.
 */

// 2d thread blocks 
#ifdef CUDA_2D_TB_KERNEL
#define TRANSLATE(x,y) (y*NODES_PER_BLOCK+x)
#ifndef CUDA_2D_FLAT
__device__ __forceinline__ void ldg_moments(CudaMultipoleMoments &m, CudaMultipoleMoments *ptr)
{
  m.radius = __ldg(&(ptr->radius));
  m.soft   = __ldg(&(ptr->soft));
  m.totalMass   = __ldg(&(ptr->totalMass));
  m.cm.x   = __ldg(&(ptr->cm.x));
  m.cm.y   = __ldg(&(ptr->cm.y));
  m.cm.z   = __ldg(&(ptr->cm.z));
#ifdef HEXADECAPOLE
  m.xx   = __ldg(&(ptr->xx));
  m.xy   = __ldg(&(ptr->xy));
  m.xz   = __ldg(&(ptr->xz));
  m.yy   = __ldg(&(ptr->yy));
  m.yz   = __ldg(&(ptr->yz));

  m.xxx   = __ldg(&(ptr->xxx));
  m.xyy   = __ldg(&(ptr->xyy));
  m.xxy   = __ldg(&(ptr->xxy));
  m.yyy   = __ldg(&(ptr->yyy));
  m.xxz   = __ldg(&(ptr->xxz));        
  m.yyz   = __ldg(&(ptr->yyz));        
  m.xyz   = __ldg(&(ptr->xyz));

  m.xxxx   = __ldg(&(ptr->xxxx));
  m.xyyy   = __ldg(&(ptr->xyyy));
  m.xxxy   = __ldg(&(ptr->xxxy));
  m.yyyy   = __ldg(&(ptr->yyyy));
  m.xxxz   = __ldg(&(ptr->xxxz));        
  m.yyyz   = __ldg(&(ptr->yyyz));        
  m.xxyy   = __ldg(&(ptr->xxyy));        
  m.xxyz   = __ldg(&(ptr->xxyz));        
  m.xyyz   = __ldg(&(ptr->xyyz));  
#else
  m.xx   = __ldg(&(ptr->xx));
  m.xy   = __ldg(&(ptr->xy));
  m.xz   = __ldg(&(ptr->xz));
  m.yy   = __ldg(&(ptr->yy));
  m.yz   = __ldg(&(ptr->yz));
  m.zz   = __ldg(&(ptr->zz));
#endif  
}

// we want to limit register usage to be 72 (by observing nvcc output)
// since GK100 has 64K registers, max threads per SM = (64K/72)
// then rounding down to multiple of 128 gives 896 
__launch_bounds__(896,1)
__global__ void nodeGravityComputation(
		CompactPartData *particleCores,
		VariablePartData *particleVars,
		CudaMultipoleMoments* moments,
		ILCell* ils,
		int *ilmarks,
		int *bucketStarts,
		int *bucketSizes,
		cudatype fperiod){
  
  // __shared__ CudaVector3D acc[THREADS_PER_BLOCK];
  // __shared__ cudatype pot[THREADS_PER_BLOCK];
  // __shared__ cudatype idt2[THREADS_PER_BLOCK];
  CudaVector3D acc;
  cudatype pot;
  cudatype idt2;
  __shared__ CudaMultipoleMoments m[NODES_PER_BLOCK];
  __shared__ int offsetID[NODES_PER_BLOCK];
  __shared__ CompactPartData shared_particle_cores[PARTS_PER_BLOCK];

  int
    start = ilmarks[blockIdx.x],
    end = ilmarks[blockIdx.x+1],
    bucketStart = bucketStarts[blockIdx.x];
  int bucketSize = bucketSizes[blockIdx.x];

  /*
  __shared__ int start;
 __shared__ int end;
 __shared__ int bucketStart;
 __shared__ int bucketSize;
 */
  
/*
  if(threadIdx.x == 0 && threadIdx.y == 0){
    start = ilmarks[blockIdx.x];
    end = ilmarks[blockIdx.x+1];
    bucketStart = bucketStarts[blockIdx.x];
    bucketSize = bucketSizes[blockIdx.x];
  }
  __syncthreads();
  */

  char
    tidx = threadIdx.x,
    tidy = threadIdx.y;

  for(int ystart = 0; ystart < bucketSize; ystart += PARTS_PER_BLOCK){
  

    int my_particle_idx = ystart + tidy;
    if(tidx == 0 && my_particle_idx < bucketSize){
      shared_particle_cores[tidy] = particleCores[bucketStart+my_particle_idx];
    }
     
    // __syncthreads(); // wait for leader threads to finish using acc's, pot's of other threads
    // acc[TRANSLATE(tidx,tidy)].x = 0.0;
    // acc[TRANSLATE(tidx,tidy)].y = 0.0;
    // acc[TRANSLATE(tidx,tidy)].z = 0.0;
    // pot[TRANSLATE(tidx,tidy)] = 0.0;
    // idt2[TRANSLATE(tidx,tidy)] = 0.0;
    acc.x = 0, acc.y = 0, acc.z = 0;
    pot = 0;
    idt2 = 0;
    
    
    for(int xstart = start; xstart < end; xstart += NODES_PER_BLOCK){
      int my_cell_idx = xstart + tidx;
      ILCell ilc;

      __syncthreads(); // wait for all threads to finish using 
                       // previous iteration's nodes before reloading
      
      if(tidy == 0 && my_cell_idx < end){
        ilc = ils[my_cell_idx];
        ldg_moments(m[tidx], &moments[ilc.index]);
        // m[tidx] = moments[ilc.index];
        offsetID[tidx] = ilc.offsetID;
      }
      
      __syncthreads(); // wait for nodes to be loaded before using them
      
      if(my_particle_idx < bucketSize && my_cell_idx < end){ // INTERACT
        CudaVector3D r;

        r.x = shared_particle_cores[tidy].position.x -
          ((((offsetID[tidx] >> 22) & 0x7)-3)*fperiod + m[tidx].cm.x);
        r.y = shared_particle_cores[tidy].position.y -
          ((((offsetID[tidx] >> 25) & 0x7)-3)*fperiod + m[tidx].cm.y);
        r.z = shared_particle_cores[tidy].position.z -
          ((((offsetID[tidx] >> 28) & 0x7)-3)*fperiod + m[tidx].cm.z);

        cudatype rsq = r.x*r.x + r.y*r.y + r.z*r.z;

        if(rsq != 0){
          cudatype dir = rsqrt(rsq);

#if defined(HEXADECAPOLE)
          // CUDA_momEvalFmomrcm(&m[tidx], &r, dir, &acc[TRANSLATE(tidx, tidy)], &pot[TRANSLATE(tidx, tidy)]);
          // idt2[TRANSLATE(tidx, tidy)] = fmax(idt2[TRANSLATE(tidx, tidy)],
          //                                (shared_particle_cores[tidy].mass + m[tidx].totalMass)*dir*dir*dir);
          CUDA_momEvalFmomrcm(&m[tidx], &r, dir, &acc, &pot);
          idt2 = fmax(idt2,
                      (shared_particle_cores[tidy].mass + m[tidx].totalMass)*dir*dir*dir);
#else
          cudatype a, b, c, d;
          cudatype
            twoh = m[tidx].soft + shared_particle_cores[tidy].soft;

          // SPLINEQ(dir, rsq, twoh, a, b, c, d);
          // expansion of function below:
          cudatype u,dih;
          if (rsq < twoh*twoh) {
            dih = 2.0/twoh;
            u = dih/dir;
            if (u < 1.0) {
              a = dih*(7.0/5.0 - 2.0/3.0*u*u + 3.0/10.0*u*u*u*u
                  - 1.0/10.0*u*u*u*u*u);
              b = dih*dih*dih*(4.0/3.0 - 6.0/5.0*u*u + 1.0/2.0*u*u*u);
              c = dih*dih*dih*dih*dih*(12.0/5.0 - 3.0/2.0*u);
              d = 3.0/2.0*dih*dih*dih*dih*dih*dih*dir;
            }
            else {
              a = -1.0/15.0*dir + dih*(8.0/5.0 - 4.0/3.0*u*u + u*u*u
                  - 3.0/10.0*u*u*u*u + 1.0/30.0*u*u*u*u*u);
              b = -1.0/15.0*dir*dir*dir + dih*dih*dih*(8.0/3.0 - 3.0*u
                  + 6.0/5.0*u*u - 1.0/6.0*u*u*u);
              c = -1.0/5.0*dir*dir*dir*dir*dir + 3.0*dih*dih*dih*dih*dir
                + dih*dih*dih*dih*dih*(-12.0/5.0 + 1.0/2.0*u);
              d = -dir*dir*dir*dir*dir*dir*dir
                + 3.0*dih*dih*dih*dih*dir*dir*dir
                - 1.0/2.0*dih*dih*dih*dih*dih*dih*dir;
            }
          }
          else {
            a = dir;
            b = a*a*a;
            c = 3.0*b*a*a;
            d = 5.0*c*a*a;
          }

          cudatype
            qirx = m[tidx].xx*r.x + m[tidx].xy*r.y + m[tidx].xz*r.z,
            qiry = m[tidx].xy*r.x + m[tidx].yy*r.y + m[tidx].yz*r.z,
            qirz = m[tidx].xz*r.x + m[tidx].yz*r.y + m[tidx].zz*r.z,
            qir = 0.5*(qirx*r.x + qiry*r.y + qirz*r.z),
            tr = 0.5*(m[tidx].xx + m[tidx].yy + m[tidx].zz),
            qir3 = b*m[tidx].totalMass + d*qir - c*tr;

          pot -= m[tidx].totalMass * a + c*qir - b*tr;
          acc.x -= qir3*r.x - c*qirx;
          acc.y -= qir3*r.y - c*qiry;
          acc.z -= qir3*r.z - c*qirz;
          idt2 = fmax(idt2, (shared_particle_cores[tidy].mass + m[tidx].totalMass)*b);

#endif
        }// end if rsq != 0
      }// end INTERACT
    }// end for each NODE group

    // __syncthreads(); // wait for all threads to finish before results become available

    cudatype sumx, sumy, sumz, poten, idt2max;
    // accumulate forces, potential in global memory data structure
    if (my_particle_idx < bucketSize) {
      sumx = acc.x, sumy = acc.y, sumz = acc.z;
      poten = pot;
      idt2max = idt2;
      for (int offset = NODES_PER_BLOCK/2; offset > 0; offset /= 2) {
        sumx += shfl_down(sumx, offset, NODES_PER_BLOCK);
        sumy += shfl_down(sumy, offset, NODES_PER_BLOCK);
        sumz += shfl_down(sumz, offset, NODES_PER_BLOCK);
        poten += shfl_down(poten, offset, NODES_PER_BLOCK);
        idt2max = fmax(idt2max, shfl_down(idt2max, offset, NODES_PER_BLOCK));
      }
      // if(tidx == 0 && my_particle_idx < bucketSize){
      if (tidx == 0) {
        // sumx = sumy = sumz = 0;
        // poten = 0;
        // idt2max = 0.0;
        // for(int i = 0; i < NODES_PER_BLOCK; i++){
        //   // sumx += acc[TRANSLATE(i,tidy)].x;
        //   // sumy += acc[TRANSLATE(i,tidy)].y;
        //   // sumz += acc[TRANSLATE(i,tidy)].z;
        //   // poten += pot[TRANSLATE(i,tidy)];
        //   idt2max = fmax(idt2[TRANSLATE(i,tidy)], idt2max);
        // }
        particleVars[bucketStart+my_particle_idx].a.x += sumx;
        particleVars[bucketStart+my_particle_idx].a.y += sumy;
        particleVars[bucketStart+my_particle_idx].a.z += sumz;
        particleVars[bucketStart+my_particle_idx].potential += poten;
        particleVars[bucketStart+my_particle_idx].dtGrav = fmax(idt2max,  particleVars[bucketStart+my_particle_idx].dtGrav);
      }
    }

  }// end for each PARTICLE group
}

#else 
__global__ void nodeGravityComputation(
		CompactPartData *particleCores,
		VariablePartData *particleVars,
		CudaMultipoleMoments *moments,
		ILCell *ils,
		int *ilmarks,
		int *bucketStarts,
		int *bucketSizes,
		cudatype fperiod){
  
  __shared__ cudatype accx[THREADS_PER_BLOCK];
  __shared__ cudatype accy[THREADS_PER_BLOCK];
  __shared__ cudatype accz[THREADS_PER_BLOCK];
  __shared__ cudatype pot[THREADS_PER_BLOCK];
  __shared__ cudatype idt2[THREADS_PER_BLOCK];
  //__shared__ cudatype mr[NODES_PER_BLOCK];
  __shared__ cudatype ms[NODES_PER_BLOCK];
  __shared__ cudatype mt[NODES_PER_BLOCK];
  __shared__ cudatype mcmx[NODES_PER_BLOCK];
  __shared__ cudatype mcmy[NODES_PER_BLOCK];
  __shared__ cudatype mcmz[NODES_PER_BLOCK];
  __shared__ cudatype mxx[NODES_PER_BLOCK];
  __shared__ cudatype mxy[NODES_PER_BLOCK];
  __shared__ cudatype mxz[NODES_PER_BLOCK];
  __shared__ cudatype myy[NODES_PER_BLOCK];
  __shared__ cudatype myz[NODES_PER_BLOCK];
  __shared__ cudatype mzz[NODES_PER_BLOCK];
  __shared__ int offsetID[NODES_PER_BLOCK];
  __shared__ CompactPartData shared_particle_cores[PARTS_PER_BLOCK];

  int start = ilmarks[blockIdx.x];
  int end = ilmarks[blockIdx.x+1];
  int bucketStart = bucketStarts[blockIdx.x];
  int bucketSize = bucketSizes[blockIdx.x];

  /*
  __shared__ int start;
 __shared__ int end;
 __shared__ int bucketStart;
 __shared__ int bucketSize;
 */

  int tx, ty;

/*
  if(threadIdx.x == 0 && threadIdx.y == 0){
    start = ilmarks[blockIdx.x];
    end = ilmarks[blockIdx.x+1];
    bucketStart = bucketStarts[blockIdx.x];
    bucketSize = bucketSizes[blockIdx.x];
  }
  __syncthreads();
  */

  int xstart;
  int ystart;
  tx = threadIdx.x;
  ty = threadIdx.y;

  for(ystart = 0; ystart < bucketSize; ystart += PARTS_PER_BLOCK){
  

    int my_particle_idx = ystart + ty;
    if(tx == 0 && my_particle_idx < bucketSize){
      shared_particle_cores[ty] = particleCores[bucketStart+my_particle_idx];
    }
    
    __syncthreads(); // wait for leader threads to finish using acc's, pot's of other threads
    accx[TRANSLATE(tx,ty)] = 0.0;
    accy[TRANSLATE(tx,ty)] = 0.0;
    accz[TRANSLATE(tx,ty)] = 0.0;
    pot[TRANSLATE(tx,ty)] = 0.0;
    idt2[TRANSLATE(tx,ty)] = 0.0;
    
    
    for(xstart = start; xstart < end; xstart += NODES_PER_BLOCK){
      int my_cell_idx = xstart + tx;
      ILCell ilc;

      __syncthreads(); // wait for all threads to finish using 
                       // previous iteration's nodes before reloading
      
      if(ty == 0 && my_cell_idx < end){
        ilc = ils[my_cell_idx];
        //mr[tx] = moments[ilc.index].radius;
        ms[tx] = moments[ilc.index].soft;
        mt[tx] = moments[ilc.index].totalMass;
        mcmx[tx] = moments[ilc.index].cm.x;
        mcmy[tx] = moments[ilc.index].cm.y;
        mcmz[tx] = moments[ilc.index].cm.z;
        mxx[tx] = moments[ilc.index].xx;
        mxy[tx] = moments[ilc.index].xy;
        mxz[tx] = moments[ilc.index].xz;
        myy[tx] = moments[ilc.index].yy;
        myz[tx] = moments[ilc.index].yz;
        mzz[tx] = moments[ilc.index].zz;
        offsetID[tx] = ilc.offsetID;
      }
      
      __syncthreads(); // wait for nodes to be loaded before using them
      
      if(my_particle_idx < bucketSize && my_cell_idx < end){ // INTERACT
        CudaVector3D r;
        cudatype rsq;
        cudatype twoh, a, b, c, d;

        r.x = shared_particle_cores[ty].position.x -
          ((((offsetID[tx] >> 22) & 0x7)-3)*fperiod + mcmx[tx]);
        r.y = shared_particle_cores[ty].position.y -
          ((((offsetID[tx] >> 25) & 0x7)-3)*fperiod + mcmy[tx]);
        r.z = shared_particle_cores[ty].position.z -
          ((((offsetID[tx] >> 28) & 0x7)-3)*fperiod + mcmz[tx]);

        rsq = r.x*r.x + r.y*r.y + r.z*r.z;        
        twoh = ms[tx] + shared_particle_cores[ty].soft;
        if(rsq != 0){
          cudatype dir = 1.0/sqrt(rsq);
          // SPLINEQ(dir, rsq, twoh, a, b, c, d);
          // expansion of function below:
          cudatype u,dih;
          if (rsq < twoh*twoh) {
            dih = 2.0/twoh;
            u = dih/dir;
            if (u < 1.0) {
              a = dih*(7.0/5.0 - 2.0/3.0*u*u + 3.0/10.0*u*u*u*u
                  - 1.0/10.0*u*u*u*u*u);
              b = dih*dih*dih*(4.0/3.0 - 6.0/5.0*u*u + 1.0/2.0*u*u*u);
              c = dih*dih*dih*dih*dih*(12.0/5.0 - 3.0/2.0*u);
              d = 3.0/2.0*dih*dih*dih*dih*dih*dih*dir;
            }
            else {
              a = -1.0/15.0*dir + dih*(8.0/5.0 - 4.0/3.0*u*u + u*u*u
                  - 3.0/10.0*u*u*u*u + 1.0/30.0*u*u*u*u*u);
              b = -1.0/15.0*dir*dir*dir + dih*dih*dih*(8.0/3.0 - 3.0*u
                  + 6.0/5.0*u*u - 1.0/6.0*u*u*u);
              c = -1.0/5.0*dir*dir*dir*dir*dir + 3.0*dih*dih*dih*dih*dir
                + dih*dih*dih*dih*dih*(-12.0/5.0 + 1.0/2.0*u);
              d = -dir*dir*dir*dir*dir*dir*dir
                + 3.0*dih*dih*dih*dih*dir*dir*dir
                - 1.0/2.0*dih*dih*dih*dih*dih*dih*dir;
            }
          }
          else {
            a = dir;
            b = a*a*a;
            c = 3.0*b*a*a;
            d = 5.0*c*a*a;
          }

          cudatype qirx = mxx[tx]*r.x + mxy[tx]*r.y + mxz[tx]*r.z;
          cudatype qiry = mxy[tx]*r.x + myy[tx]*r.y + myz[tx]*r.z;
          cudatype qirz = mxz[tx]*r.x + myz[tx]*r.y + mzz[tx]*r.z;
          cudatype qir = 0.5*(qirx*r.x + qiry*r.y + qirz*r.z);
          cudatype tr = 0.5*(mxx[tx] + myy[tx] + mzz[tx]);
          cudatype qir3 = b*mt[tx] + d*qir - c*tr;

          pot[TRANSLATE(tx, ty)] -= mt[tx] * a + c*qir - b*tr;

          accx[TRANSLATE(tx, ty)] -= qir3*r.x - c*qirx;
          accy[TRANSLATE(tx, ty)] -= qir3*r.y - c*qiry;
          accz[TRANSLATE(tx, ty)] -= qir3*r.z - c*qirz;
          idt2[TRANSLATE(tx, ty)] = fmax(idt2[TRANSLATE(tx, ty)],
                                    (shared_particle_cores[ty].mass + mt[tx]) * b);
        }// end if rsq != 0
      }// end INTERACT
    }// end for each NODE group

    __syncthreads(); // wait for all threads to finish before results become available

    cudatype sumx, sumy, sumz, poten, idt2max;
    sumx = sumy = sumz = poten = idt2max = 0.0;
    // accumulate forces, potential in global memory data structure
    if(tx == 0 && my_particle_idx < bucketSize){
      for(int i = 0; i < NODES_PER_BLOCK; i++){
        sumx += accx[TRANSLATE(i,ty)];
        sumy += accy[TRANSLATE(i,ty)];
        sumz += accz[TRANSLATE(i,ty)];
        poten += pot[TRANSLATE(i,ty)];
        idt2max = fmax(idt2[TRANSLATE(i,ty)], idt2max);
      }
      particleVars[bucketStart+my_particle_idx].a.x += sumx;
      particleVars[bucketStart+my_particle_idx].a.y += sumy;
      particleVars[bucketStart+my_particle_idx].a.z += sumz;
      particleVars[bucketStart+my_particle_idx].potential += poten;
      particleVars[bucketStart+my_particle_idx].dtGrav = fmax(idt2max,  particleVars[bucketStart+my_particle_idx].dtGrav);
    }

  }// end for each PARTICLE group
}
#endif
#else
__global__ void nodeGravityComputation(
		CompactPartData *particleCores,
		VariablePartData *particleVars,
		CudaMultipoleMoments *moments,
		ILCell *ils,
		int *ilmarks,
		int *bucketStarts,
		int *bucketSizes,
		cudatype fperiod){

  // each thread has its own storage for these
  __shared__ CudaVector3D acc[THREADS_PER_BLOCK];
  __shared__ cudatype pot[THREADS_PER_BLOCK];
  __shared__ cudatype idt2[THREADS_PER_BLOCK];
  __shared__ CudaMultipoleMoments m[THREADS_PER_BLOCK];

  __shared__ CompactPartData shared_particle_core;


  // each block is given a bucket to compute
  // each thread in the block computes an interaction of a particle with a node
  // threads must iterate through the interaction lists and sync.
  // then, block leader (first rank in each block) reduces the forces and commits 
  // values to global memory.
  int bucket = blockIdx.x;
  int start = ilmarks[bucket];
  int end = ilmarks[bucket+1];
  int bucketSize = bucketSizes[bucket];
  int bucketStart = bucketStarts[bucket];
  int thread = threadIdx.x;

  CudaVector3D r;
  cudatype rsq;
  cudatype twoh, a, b, c, d;

  for(int particle = 0; particle < bucketSize; particle++){
    if(thread == 0){
      // load shared_particle_core
      shared_particle_core = particleCores[bucketStart+particle];
    }
    __syncthreads();

    acc[thread].x = 0;
    acc[thread].y = 0;
    acc[thread].z = 0;
    pot[thread] = 0;
    idt2[thread] = 0;

    for(int node = start+thread; node < end; node+=THREADS_PER_BLOCK){
      ILCell ilc = ils[node];
      m[thread] = moments[ilc.index];
      int offsetID = ilc.offsetID;

      r.x = shared_particle_core.position.x -
        ((((offsetID >> 22) & 0x7)-3)*fperiod + m[thread].cm.x);
      r.y = shared_particle_core.position.y -
        ((((offsetID >> 25) & 0x7)-3)*fperiod + m[thread].cm.y);
      r.z = shared_particle_core.position.z -
        ((((offsetID >> 28) & 0x7)-3)*fperiod + m[thread].cm.z);

      rsq = r.x*r.x + r.y*r.y + r.z*r.z;        
      twoh = m[thread].soft + shared_particle_core.soft;
      if(rsq != 0){
        cudatype dir = 1.0/sqrt(rsq);
        // SPLINEQ(dir, rsq, twoh, a, b, c, d);
        // expansion of function below:
        cudatype u,dih;
        if (rsq < twoh*twoh) {
          dih = 2.0/twoh;
          u = dih/dir;
          if (u < 1.0) {
            a = dih*(7.0/5.0 - 2.0/3.0*u*u + 3.0/10.0*u*u*u*u
                - 1.0/10.0*u*u*u*u*u);
            b = dih*dih*dih*(4.0/3.0 - 6.0/5.0*u*u + 1.0/2.0*u*u*u);
            c = dih*dih*dih*dih*dih*(12.0/5.0 - 3.0/2.0*u);
            d = 3.0/2.0*dih*dih*dih*dih*dih*dih*dir;
          }
          else {
            a = -1.0/15.0*dir + dih*(8.0/5.0 - 4.0/3.0*u*u + u*u*u
                - 3.0/10.0*u*u*u*u + 1.0/30.0*u*u*u*u*u);
            b = -1.0/15.0*dir*dir*dir + dih*dih*dih*(8.0/3.0 - 3.0*u
                + 6.0/5.0*u*u - 1.0/6.0*u*u*u);
            c = -1.0/5.0*dir*dir*dir*dir*dir + 3.0*dih*dih*dih*dih*dir
              + dih*dih*dih*dih*dih*(-12.0/5.0 + 1.0/2.0*u);
            d = -dir*dir*dir*dir*dir*dir*dir
              + 3.0*dih*dih*dih*dih*dir*dir*dir
              - 1.0/2.0*dih*dih*dih*dih*dih*dih*dir;
          }
        }
        else {
          a = dir;
          b = a*a*a;
          c = 3.0*b*a*a;
          d = 5.0*c*a*a;
        }

        cudatype qirx = m[thread].xx*r.x + m[thread].xy*r.y + m[thread].xz*r.z;
        cudatype qiry = m[thread].xy*r.x + m[thread].yy*r.y + m[thread].yz*r.z;
        cudatype qirz = m[thread].xz*r.x + m[thread].yz*r.y + m[thread].zz*r.z;
        cudatype qir = 0.5*(qirx*r.x + qiry*r.y + qirz*r.z);
        cudatype tr = 0.5*(m[thread].xx + m[thread].yy + m[thread].zz);
        cudatype qir3 = b*m[thread].totalMass + d*qir - c*tr;

        pot[thread] -= m[thread].totalMass * a + c*qir - b*tr;

        acc[thread].x -= qir3*r.x - c*qirx;
        acc[thread].y -= qir3*r.y - c*qiry;
        acc[thread].z -= qir3*r.z - c*qirz;
        idt2[thread] = fmax(idt2[thread], (shared_particle_core.mass + m[thread].totalMass)*b);
      }// end if rsq != 0
    }// for each node in list
    __syncthreads();
    // at this point, the total force on particle is distributed among
    // all active threads;
    // reduce.
    // TODO: make this a parallel reduction

    cudatype sumx, sumy, sumz, poten, idt2max;
    sumx = sumy = sumz = poten = idt2max = 0.0;
    if(thread == 0){
      for(int i = 0; i < THREADS_PER_BLOCK; i++){
        sumx += acc[i].x;
        sumy += acc[i].y;
        sumz += acc[i].z;
        poten += pot[i];
        idt2max = fmax(idt2[i], idt2max);
      }
      particleVars[bucketStart+particle].a.x += sumx;
      particleVars[bucketStart+particle].a.y += sumy;
      particleVars[bucketStart+particle].a.z += sumz;
      particleVars[bucketStart+particle].potential += poten;
      particleVars[bucketStart+particle].dtGrav = fmax(idt2max,  particleVars[bucketStart+particle].dtGrav);
    }
  }// for each particle in bucket

      
}
#endif

/**
 * @brief interaction between source particles and buckets of particles.
 * @param particleCores Read-only properties of target particles.
 * @param particleVars Accumulators of accelerations etc. of target particles.
 * @param sourceCores Properties of source particles.
 * @param ils array of "cells": index into sourceCores and offset for each particle.
 * @param ilmarks Indices into ils for each block.
 * @param bucketStarts Indices into particleCores and particleVars
 *                      for each block
 * @param bucketSizes Size of the bucket for each block
 * @param fPeriod Size of periodic boundary condition.
 */
__device__ __forceinline__ void ldg_cPartData(CompactPartData &m, CompactPartData *ptr)
{
  m.mass         = __ldg(&(ptr->mass));
  m.soft         = __ldg(&(ptr->soft));
  m.position.x   = __ldg(&(ptr->position.x));
  m.position.y   = __ldg(&(ptr->position.y));
  m.position.z   = __ldg(&(ptr->position.z));
}

#ifdef CUDA_2D_TB_KERNEL
#define TRANSLATE_PART(x,y) (y*NODES_PER_BLOCK_PART+x)
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
//maxThreadsPerBlock needs to be a multiple of 128, this can be used 
//to limits the number of  register per thread. More threads less 
//registers  
__launch_bounds__(1408, 1)
__global__ void particleGravityComputation(
		CompactPartData *targetCores,
		VariablePartData *targetVars,
		CompactPartData* sourceCores,
		ILCell *ils,
		int *ilmarks,
		int *bucketStarts,
		int *bucketSizes,
		cudatype fperiod){
  
  //__shared__ CudaVector3D acc[THREADS_PER_BLOCK_PART];
  // __shared__ cudatype pot[THREADS_PER_BLOCK_PART];
  // __shared__ cudatype idt2[THREADS_PER_BLOCK_PART];
  CudaVector3D acc;
  cudatype pot;
  cudatype idt2;
  __shared__ CompactPartData m[NODES_PER_BLOCK_PART];
  __shared__ int offsetID[NODES_PER_BLOCK_PART];
  __shared__ CompactPartData shared_particle_cores[PARTS_PER_BLOCK_PART];

  int start = ilmarks[blockIdx.x];
  int end = ilmarks[blockIdx.x+1];
  int bucketStart = bucketStarts[blockIdx.x];
  int bucketSize = bucketSizes[blockIdx.x];

  /*
  __shared__ int start;
 __shared__ int end;
 __shared__ int bucketStart;
 __shared__ int bucketSize;
 */

  int tx, ty;

/*
  if(threadIdx.x == 0 && threadIdx.y == 0){
    start = ilmarks[blockIdx.x];
    end = ilmarks[blockIdx.x+1];
    bucketStart = bucketStarts[blockIdx.x];
    bucketSize = bucketSizes[blockIdx.x];
  }
  __syncthreads();
  */

  int xstart;
  int ystart;
  tx = threadIdx.x;
  ty = threadIdx.y;

  for(ystart = 0; ystart < bucketSize; ystart += PARTS_PER_BLOCK_PART){
  

    int my_particle_idx = ystart + ty;
    if(tx == 0 && my_particle_idx < bucketSize){
      shared_particle_cores[ty] = targetCores[bucketStart+my_particle_idx];
    }
    
    //__syncthreads(); // wait for leader threads to finish using acc's, pot's of other threads
    //acc[TRANSLATE_PART(tx,ty)].x = 0.0;
    //acc[TRANSLATE_PART(tx,ty)].y = 0.0;
    //acc[TRANSLATE_PART(tx,ty)].z = 0.0;
    //pot[TRANSLATE_PART(tx,ty)] = 0.0;
    //idt2[TRANSLATE_PART(tx,ty)] = 0.0;
    acc.x = 0, acc.y = 0, acc.z = 0;
    pot = 0;
    idt2 = 0;
    
    
    for(xstart = start; xstart < end; xstart += NODES_PER_BLOCK_PART){
      int my_cell_idx = xstart + tx;
      ILCell ilc;

      __syncthreads(); // wait for all threads to finish using 
                       // previous iteration's nodes before reloading
      
      if(ty == 0 && my_cell_idx < end){
        ilc = ils[my_cell_idx];
        ldg_cPartData(m[tx], &sourceCores[ilc.index]);
        //m[tx] = sourceCores[ilc.index];
        offsetID[tx] = ilc.offsetID;
      }
      
      __syncthreads(); // wait for nodes to be loaded before using them
      
      if(my_particle_idx < bucketSize && my_cell_idx < end){ // INTERACT
        CudaVector3D r;
        cudatype rsq;
        cudatype twoh, a, b;

        r.x = (((offsetID[tx] >> 22) & 0x7)-3)*fperiod 
              + m[tx].position.x
              - shared_particle_cores[ty].position.x;
        r.y = (((offsetID[tx] >> 25) & 0x7)-3)*fperiod 
              + m[tx].position.y
              - shared_particle_cores[ty].position.y;
        r.z = (((offsetID[tx] >> 28) & 0x7)-3)*fperiod 
              + m[tx].position.z
              - shared_particle_cores[ty].position.z;

        rsq = r.x*r.x + r.y*r.y + r.z*r.z;        
        twoh = m[tx].soft + shared_particle_cores[ty].soft;
        if(rsq != 0){
          cudatype r1, u, dih, dir;
          r1 = sqrt(rsq);
          if (r1 < (twoh)) {
            dih = 2.0/(twoh);
            u = r1*dih;
            if (u < 1.0) {
              a = dih*(7.0/5.0 - 2.0/3.0*u*u + 3.0/10.0*u*u*u*u
                  - 1.0/10.0*u*u*u*u*u);
              b = dih*dih*dih*(4.0/3.0 - 6.0/5.0*u*u
                  + 1.0/2.0*u*u*u);
            }
            else {
              dir = 1.0/r1;
              a = -1.0/15.0*dir + dih*(8.0/5.0 - 4.0/3.0*u*u +
                  u*u*u - 3.0/10.0*u*u*u*u + 1.0/30.0*u*u*u*u*u);
              b = -1.0/15.0*dir*dir*dir +
                dih*dih*dih*(8.0/3.0 - 3.0*u +
                    6.0/5.0*u*u - 1.0/6.0*u*u*u);
            }
          }
          else {
            a = 1.0/r1;
            b = a*a*a;
          }

          //pot[TRANSLATE_PART(tx, ty)] -= m[tx].mass * a;	

          //acc[TRANSLATE_PART(tx, ty)].x += r.x*b*m[tx].mass;
          //acc[TRANSLATE_PART(tx, ty)].y += r.y*b*m[tx].mass;
          //acc[TRANSLATE_PART(tx, ty)].z += r.z*b*m[tx].mass;
          //idt2[TRANSLATE_PART(tx, ty)] = fmax(idt2[TRANSLATE_PART(tx, ty)],
          //                               (shared_particle_cores[ty].mass + m[tx].mass) * b);

	  pot -= m[tx].mass * a;

          acc.x += r.x*b*m[tx].mass;
          acc.y += r.y*b*m[tx].mass;
          acc.z += r.z*b*m[tx].mass;
          idt2 = fmax(idt2, (shared_particle_cores[ty].mass + m[tx].mass) * b);

        }// end if rsq != 0
      }// end INTERACT
    }// end for each NODE group

    //__syncthreads(); // wait for all threads to finish before results become available

    cudatype sumx, sumy, sumz, poten, idt2max;
    sumx = sumy = sumz = poten = idt2max = 0.0;
    // accumulate forces, potential in global memory data structure
    //if(tx == 0 && my_particle_idx < bucketSize){
    //  for(int i = 0; i < NODES_PER_BLOCK_PART; i++){
    //    sumx += acc[TRANSLATE_PART(i,ty)].x;
    //    sumy += acc[TRANSLATE_PART(i,ty)].y;
    //    sumz += acc[TRANSLATE_PART(i,ty)].z;
    //    poten += pot[TRANSLATE_PART(i,ty)];
    //    idt2max = fmax(idt2[TRANSLATE_PART(i,ty)], idt2max);
    //  }

    // accumulate forces, potential in global memory data structure
    if(my_particle_idx < bucketSize){
      sumx = acc.x;
      sumy = acc.y; 
      sumz = acc.z;
      poten = pot;
      idt2max = idt2;
      for(int offset = NODES_PER_BLOCK/2; offset > 0; offset /= 2){
        sumx += shfl_down(sumx, offset, NODES_PER_BLOCK_PART);
        sumy += shfl_down(sumy, offset, NODES_PER_BLOCK_PART);
        sumz += shfl_down(sumz, offset, NODES_PER_BLOCK_PART);
        poten += shfl_down(poten, offset, NODES_PER_BLOCK_PART);
        idt2max = fmax(idt2max, shfl_down(idt2max, offset, NODES_PER_BLOCK_PART));
      }

      if(tx == 0){
      	targetVars[bucketStart+my_particle_idx].a.x += sumx;
      	targetVars[bucketStart+my_particle_idx].a.y += sumy;
     	targetVars[bucketStart+my_particle_idx].a.z += sumz;
      	targetVars[bucketStart+my_particle_idx].potential += poten;
      	targetVars[bucketStart+my_particle_idx].dtGrav = fmax(idt2max,  targetVars[bucketStart+my_particle_idx].dtGrav);
      }
    }

  }// end for each PARTICLE group
}
#else
__launch_bounds__(896, 1)
__global__ void particleGravityComputation(
                                   CompactPartData *targetCores,
                                   VariablePartData *targetVars,
                                   CompactPartData *sourceCores,
                                   ILPart *ils,
                                   int *ilmarks,
		                   int *bucketStarts,
		                   int *bucketSizes,
		                   cudatype fperiod){

  // each thread has its own storage for these
  __shared__ CudaVector3D acc[THREADS_PER_BLOCK];
  __shared__ cudatype pot[THREADS_PER_BLOCK];
  __shared__ cudatype idt2[THREADS_PER_BLOCK];
  __shared__ CompactPartData source_cores[THREADS_PER_BLOCK];

  __shared__ CompactPartData shared_target_core;


  // each block is given a bucket to compute
  // each thread in the block computes an interaction of a particle with a node
  // threads must iterate through the interaction lists and sync.
  // then, block leader (first rank in each block) reduces the forces and commits 
  // values to global memory.
  int bucket = blockIdx.x;
  int start = ilmarks[bucket];
  int end = ilmarks[bucket+1];
  int bucketSize = bucketSizes[bucket];
  int bucketStart = bucketStarts[bucket];
  int thread = threadIdx.x;

  CudaVector3D r;
  cudatype rsq;
  cudatype twoh, a, b;

  for(int target = 0; target < bucketSize; target++){
    if(thread == 0){
      shared_target_core = targetCores[bucketStart+target];
    }
    __syncthreads();

    acc[thread].x = 0;
    acc[thread].y = 0;
    acc[thread].z = 0;
    pot[thread] = 0;
    idt2[thread] = 0;

    for(int source = start+thread; source < end; source += THREADS_PER_BLOCK){
      ILPart ilp = ils[source]; 
      int oid = ilp.off;
      int num = ilp.num;
      int ilpindex = ilp.index;

      for(int particle = 0; particle < num; particle++){
        source_cores[thread] = sourceCores[ilpindex+particle];

        r.x = (((oid >> 22) & 0x7)-3)*fperiod +
          source_cores[thread].position.x -
          shared_target_core.position.x;

        r.y = (((oid >> 25) & 0x7)-3)*fperiod +
          source_cores[thread].position.y -
          shared_target_core.position.y;

        r.z = (((oid >> 28) & 0x7)-3)*fperiod +
          source_cores[thread].position.z -
          shared_target_core.position.z;

        rsq = r.x*r.x + r.y*r.y + r.z*r.z;
        twoh = source_cores[thread].soft + shared_target_core.soft;
        if(rsq != 0){
          cudatype r1, u,dih,dir;
          r1 = sqrt(rsq);
          if (r1 < (twoh)) {
            dih = 2.0/(twoh);
            u = r1*dih;
            if (u < 1.0) {
              a = dih*(7.0/5.0 - 2.0/3.0*u*u + 3.0/10.0*u*u*u*u
                  - 1.0/10.0*u*u*u*u*u);
              b = dih*dih*dih*(4.0/3.0 - 6.0/5.0*u*u
                  + 1.0/2.0*u*u*u);
            }
            else {
              dir = 1.0/r1;
              a = -1.0/15.0*dir + dih*(8.0/5.0 - 4.0/3.0*u*u +
                  u*u*u - 3.0/10.0*u*u*u*u + 1.0/30.0*u*u*u*u*u);
              b = -1.0/15.0*dir*dir*dir +
                dih*dih*dih*(8.0/3.0 - 3.0*u +
                    6.0/5.0*u*u - 1.0/6.0*u*u*u);
            }
          }
          else {
            a = 1.0/r1;
            b = a*a*a;
          }

          pot[thread] -= source_cores[thread].mass * a;

          acc[thread].x += r.x*b*source_cores[thread].mass;
          acc[thread].y += r.y*b*source_cores[thread].mass;
          acc[thread].z += r.z*b*source_cores[thread].mass;
          idt2[thread] = fmax(idt2[thread], (shared_target_core.mass + source_cores[thread].mass) * b);
        }// if rsq != 0
      }// for each particle in source bucket
    }// for each source bucket 

    __syncthreads();
  
    cudatype sumx, sumy, sumz, poten, idt2max;
    sumx = sumy = sumz = poten = idt2max = 0.0;
    if(thread == 0){
      for(int i = 0; i < THREADS_PER_BLOCK; i++){
        sumx += acc[i].x;
        sumy += acc[i].y;
        sumz += acc[i].z;
        poten += pot[i];
        idt2max = fmax(idt2[i], idt2max);
      }
      targetVars[bucketStart+target].a.x += sumx;
      targetVars[bucketStart+target].a.y += sumy;
      targetVars[bucketStart+target].a.z += sumz;
      targetVars[bucketStart+target].potential += poten;
      targetVars[bucketStart+target].dtGrav = fmax(idt2max,  targetVars[bucketStart+target].dtGrav);
    }

  }// for each target part
}
#endif

__global__ void EwaldKernel(CompactPartData *particleCores, VariablePartData *particleVars, int *markers, int largephase, int First, int Last);

extern unsigned int timerHandle; 

void EwaldHostMemorySetup(EwaldData *h_idata, int nParticles, int nEwhLoop, int largephase) {
  if(largephase)
    allocatePinnedHostMemory((void **)&(h_idata->EwaldMarkers), nParticles*sizeof(int));
  else
    h_idata->EwaldMarkers = NULL;
  allocatePinnedHostMemory((void **)&(h_idata->ewt), nEwhLoop*sizeof(EwtData));
  allocatePinnedHostMemory((void **)&(h_idata->cachedData), sizeof(EwaldReadOnlyData));
}

void EwaldHostMemoryFree(EwaldData *h_idata, int largephase) {
  if(largephase)
    freePinnedHostMemory(h_idata->EwaldMarkers);
  freePinnedHostMemory(h_idata->ewt);
  freePinnedHostMemory(h_idata->cachedData);
}

/** @brief Set up CUDA kernels to perform Ewald sum.
 *  @param d_localParts Local particle data on device
 *  @param d_localVars Local particle accelerations on device
 *  @param h_idata Host data buffers
 *  @param stream CUDA stream to perform GPU operations over
 *  @param cb Callback
 *  @param myIndex Chare index on this node that called this request.
 *  @param largephase Whether to perform large or small phase calculation
 *  
 *  The "top" and "bottom" Ewlad kernels have been combined:
 *    "top" for the real space loop,
 *    "bottom" for the k-space loop.
 *  
 */
void EwaldHost(CompactPartData *d_localParts, VariablePartData *d_localVars,
               EwaldData *h_idata, cudaStream_t stream, void *cb, int myIndex, int largephase)
{
  int n = h_idata->cachedData->n;
  int numBlocks = (int) ceilf((float)n/BLOCK_SIZE);
  int nEwhLoop = h_idata->cachedData->nEwhLoop;
  assert(nEwhLoop <= NEWH);

  size_t size;
  if(largephase) size = n * sizeof(int);
  else size = 0;

  HAPI_TRACE_BEGIN();
  int *d_EwaldMarkers;
  cudaChk(cudaMalloc(&d_EwaldMarkers, size));

  cudaMemcpyAsync(d_EwaldMarkers, h_idata->EwaldMarkers, size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(cachedData, h_idata->cachedData, sizeof(EwaldReadOnlyData), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(ewt, h_idata->ewt, nEwhLoop * sizeof(EwtData), 0, cudaMemcpyHostToDevice, stream);

  if (largephase)
      EwaldKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_localParts, 
		                                         d_localVars,
							 d_EwaldMarkers, 1,
							 h_idata->EwaldRange[0], h_idata->EwaldRange[1]);
  else
      EwaldKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_localParts, 
                                                         d_localVars,
							 NULL, 0,
							 h_idata->EwaldRange[0], h_idata->EwaldRange[1]);
  HAPI_TRACE_END(CUDA_EWALD);

#ifdef CUDA_VERBOSE_KERNEL_ENQUEUE
  printf("[%d] in EwaldHost, enqueued EwaldKernel\n", myIndex);
#endif

  cudaChk(cudaPeekAtLastError());
  hapiAddCallback(stream, cb);
  cudaChk(cudaFree(d_EwaldMarkers));
}

__global__ void EwaldKernel(CompactPartData *particleCores, 
                               VariablePartData *particleVars, 
                               int *markers, int largephase,
                               int First, int Last) {
  /////////////////////////////////////
  ////////////// Ewald TOP ////////////
  /////////////////////////////////////
  int id;
  if(largephase){
    id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(id > Last) return;
    id = markers[id];
  }else{
    id = First + blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(id > Last) return;
  }

  CompactPartData *p;

  cudatype alphan;
  cudatype fPot, ax, ay, az;
  cudatype x, y, z, r2, dir, dir2, a; 
  cudatype xdif, ydif, zdif; 
  cudatype g0, g1, g2, g3;
  cudatype Q2, Q2mirx, Q2miry, Q2mirz, Q2mir, Qta; 
  int ix, iy, iz, bInHole, bInHolex, bInHolexy;

#ifdef HEXADECAPOLE
  MomcData *mom = &(cachedData->momcRoot);
  MultipoleMomentsData *momQuad = &(cachedData->mm);
  cudatype xx,xxx,xxy,xxz,yy,yyy,yyz,xyy,zz,zzz,xzz,yzz,xy,xyz,xz,yz;
  cudatype g4, g5;
  cudatype Q4mirx,Q4miry,Q4mirz,Q4mir,Q4x,Q4y,Q4z;
  cudatype Q4xx,Q4xy,Q4xz,Q4yy,Q4yz,Q4zz,Q4,Q3x,Q3y,Q3z;
  cudatype Q3mirx,Q3miry,Q3mirz,Q3mir;
  const cudatype onethird = 1.0/3.0;
#else
  MultipoleMomentsData *mom;
  mom = &(cachedData->mm);
#endif

#ifdef HEXADECAPOLE
  Q4xx = 0.5*(mom->xxxx + mom->xxyy + mom->xxzz);
  Q4xy = 0.5*(mom->xxxy + mom->xyyy + mom->xyzz);
  Q4xz = 0.5*(mom->xxxz + mom->xyyz + mom->xzzz);
  Q4yy = 0.5*(mom->xxyy + mom->yyyy + mom->yyzz);
  Q4yz = 0.5*(mom->xxyz + mom->yyyz + mom->yzzz);
  Q4zz = 0.5*(mom->xxzz + mom->yyzz + mom->zzzz);
  Q4 = 0.25*(Q4xx + Q4yy + Q4zz);
  Q3x = 0.5*(mom->xxx + mom->xyy + mom->xzz);
  Q3y = 0.5*(mom->xxy + mom->yyy + mom->yzz);
  Q3z = 0.5*(mom->xxz + mom->yyz + mom->zzz);
#endif

  Q2 = 0.5 * (mom->xx + mom->yy + mom->zz);

  p = &(particleCores[id]);

#ifdef DEBUG
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Moments\n");
    printf("xx %f, xy %f, xz %f, yy %f, yz %f, zz %f\n", mom->xx, mom->xy, mom->xz,
        mom->yy, mom->yz, mom->zz);
  }
#endif

  ax = 0.0f;
  ay = 0.0f;
  az = 0.0f;

#ifdef HEXADECAPOLE
  xdif = p->position.x - momQuad->cmx; 
  ydif = p->position.y - momQuad->cmy; 
  zdif = p->position.z - momQuad->cmz;
  fPot = momQuad->totalMass*cachedData->k1;
#else
  xdif = p->position.x - mom->cmx; 
  ydif = p->position.y - mom->cmy; 
  zdif = p->position.z - mom->cmz;
  fPot = mom->totalMass*cachedData->k1;
#endif
  for (ix=-(cachedData->nEwReps);ix<=(cachedData->nEwReps);++ix) {  
    bInHolex = (ix >= -cachedData->nReps && ix <= cachedData->nReps);
    x = xdif + ix * cachedData->L;
    for(iy=-(cachedData->nEwReps);iy<=(cachedData->nEwReps);++iy) {
      bInHolexy = (bInHolex && iy >= -cachedData->nReps && iy <= cachedData->nReps);
      y = ydif + iy*cachedData->L;
      for(iz=-(cachedData->nEwReps);iz<=(cachedData->nEwReps);++iz) {
        bInHole = (bInHolexy && iz >= -cachedData->nReps && iz <= cachedData->nReps);
        z = zdif + iz*cachedData->L;
        r2 = x*x + y*y + z*z;
        if (r2 > cachedData->fEwCut2 && !bInHole) continue;
        if (r2 < cachedData->fInner2) {

          /*
           * For small r, series expand about
           * the origin to avoid errors caused
           * by cancellation of large terms.
           * N.B. The following uses expf(), erfcf(), etc. If these ever
           * get changed to use double precision, then fInner2 also needs
           * to be changed. See line 480 in Ewald.cpp.
           */

          alphan = cachedData->ka;
          r2 *= cachedData->alpha2;
          g0 = alphan*((1.0/3.0)*r2 - 1.0);
          alphan *= 2*cachedData->alpha2;
          g1 = alphan*((1.0/5.0)*r2 - (1.0/3.0));
          alphan *= 2*cachedData->alpha2;
          g2 = alphan*((1.0/7.0)*r2 - (1.0/5.0));
          alphan *= 2*cachedData->alpha2;
          g3 = alphan*((1.0/9.0)*r2 - (1.0/7.0));
#ifdef HEXADECAPOLE
	  alphan *= 2*cachedData->alpha2;
	  g4 = alphan*((1.0/11.0)*r2 - (1.0/9.0));
	  alphan *= 2*cachedData->alpha2;
	  g5 = alphan*((1.0/13.0)*r2 - (1.0/11.0));
#endif
        }
        else {
          dir = 1/sqrtf(r2);
          dir2 = dir*dir;
          a = expf(-r2*cachedData->alpha2);
          a *= cachedData->ka*dir2;
          if (bInHole) g0 = -erff(cachedData->alpha/dir);
          else g0 = erfcf(cachedData->alpha/dir);
          g0 *= dir;
          g1 = g0*dir2 + a;
          alphan = 2*cachedData->alpha2;
          g2 = 3*g1*dir2 + alphan*a;
          alphan *= 2*cachedData->alpha2;
          g3 = 5*g2*dir2 + alphan*a;
#ifdef HEXADECAPOLE
	  alphan *= 2*cachedData->alpha2;
	  g4 = 7*g3*dir2 + alphan*a;
	  alphan *= 2*cachedData->alpha2;
	  g5 = 9*g4*dir2 + alphan*a;
#endif
        }
#ifdef HEXADECAPOLE
	xx = 0.5*x*x;
	xxx = onethird*xx*x;
	xxy = xx*y;
	xxz = xx*z;
	yy = 0.5*y*y;
	yyy = onethird*yy*y;
	xyy = yy*x;
	yyz = yy*z;
	zz = 0.5*z*z;
	zzz = onethird*zz*z;
	xzz = zz*x;
	yzz = zz*y;
	xy = x*y;
	xyz = xy*z;
	xz = x*z;
	yz = y*z;
	Q2mirx = mom->xx*x + mom->xy*y + mom->xz*z;
	Q2miry = mom->xy*x + mom->yy*y + mom->yz*z;
	Q2mirz = mom->xz*x + mom->yz*y + mom->zz*z;
	Q3mirx = mom->xxx*xx + mom->xxy*xy + mom->xxz*xz + mom->xyy*yy + mom->xyz*yz + mom->xzz*zz;
	Q3miry = mom->xxy*xx + mom->xyy*xy + mom->xyz*xz + mom->yyy*yy + mom->yyz*yz + mom->yzz*zz;
	Q3mirz = mom->xxz*xx + mom->xyz*xy + mom->xzz*xz + mom->yyz*yy + mom->yzz*yz + mom->zzz*zz;
	Q4mirx = mom->xxxx*xxx + mom->xxxy*xxy + mom->xxxz*xxz + mom->xxyy*xyy + mom->xxyz*xyz +
	  mom->xxzz*xzz + mom->xyyy*yyy + mom->xyyz*yyz + mom->xyzz*yzz + mom->xzzz*zzz;
	Q4miry = mom->xxxy*xxx + mom->xxyy*xxy + mom->xxyz*xxz + mom->xyyy*xyy + mom->xyyz*xyz +
	  mom->xyzz*xzz + mom->yyyy*yyy + mom->yyyz*yyz + mom->yyzz*yzz + mom->yzzz*zzz;
	Q4mirz = mom->xxxz*xxx + mom->xxyz*xxy + mom->xxzz*xxz + mom->xyyz*xyy + mom->xyzz*xyz +
	  mom->xzzz*xzz + mom->yyyz*yyy + mom->yyzz*yyz + mom->yzzz*yzz + mom->zzzz*zzz;
	Q4x = Q4xx*x + Q4xy*y + Q4xz*z;
	Q4y = Q4xy*x + Q4yy*y + Q4yz*z;
	Q4z = Q4xz*x + Q4yz*y + Q4zz*z;
	Q2mir = 0.5*(Q2mirx*x + Q2miry*y + Q2mirz*z) - (Q3x*x + Q3y*y + Q3z*z) + Q4;
	Q3mir = onethird*(Q3mirx*x + Q3miry*y + Q3mirz*z) - 0.5*(Q4x*x + Q4y*y + Q4z*z);
	Q4mir = 0.25*(Q4mirx*x + Q4miry*y + Q4mirz*z);
	Qta = g1*mom->m - g2*Q2 + g3*Q2mir + g4*Q3mir + g5*Q4mir;
	fPot -= g0*mom->m - g1*Q2 + g2*Q2mir + g3*Q3mir + g4*Q4mir;
	ax += g2*(Q2mirx - Q3x) + g3*(Q3mirx - Q4x) + g4*Q4mirx - x*Qta;
	ay += g2*(Q2miry - Q3y) + g3*(Q3miry - Q4y) + g4*Q4miry - y*Qta;
	az += g2*(Q2mirz - Q3z) + g3*(Q3mirz - Q4z) + g4*Q4mirz - z*Qta;
#else
        Q2mirx = mom->xx*x + mom->xy*y + mom->xz*z;
        Q2miry = mom->xy*x + mom->yy*y + mom->yz*z;
        Q2mirz = mom->xz*x + mom->yz*y + mom->zz*z;
        Q2mir = 0.5*(Q2mirx*x + Q2miry*y + Q2mirz*z);
        Qta = g1*mom->totalMass - g2*Q2 + g3*Q2mir;
        fPot -= g0*mom->totalMass - g1*Q2 + g2*Q2mir;

        ax += g2*(Q2mirx) - x*Qta;
        ay += g2*(Q2miry) - y*Qta;
        az += g2*(Q2mirz) - z*Qta;

#endif
      }
    }
  }

  /////////////////////////////////////
  //////////// Ewald Bottom ///////////
  /////////////////////////////////////
  cudatype hdotx, c, s;
  cudatype tempEwt; 
  xdif = ydif = zdif = 0.0; 

  MultipoleMomentsData *momBottom = &(cachedData->mm);

  /*
   ** Scoring for the h-loop (+,*)
   **        Without trig = (10,14)
   **          Trig est.    = 2*(6,11)  same as 1/sqrt scoring.
   **            Total        = (22,36)
   **                                   = 58
   */

  xdif = p->position.x - momBottom->cmx; 
  ydif = p->position.y - momBottom->cmy; 
  zdif = p->position.z - momBottom->cmz; 

  for (int i=0;i<cachedData->nEwhLoop;++i) {
    hdotx = ewt[i].hx * xdif + ewt[i].hy * ydif + ewt[i].hz * zdif;
    c = cosf(hdotx);
    s = sinf(hdotx);    
    fPot += ewt[i].hCfac*c + ewt[i].hSfac*s;    
    tempEwt = ewt[i].hCfac*s - ewt[i].hSfac*c;
    ax += ewt[i].hx * tempEwt;
    ay += ewt[i].hy * tempEwt;
    az += ewt[i].hz * tempEwt;
  }

  particleVars[id].a.x += ax;
  particleVars[id].a.y += ay;
  particleVars[id].a.z += az;
  particleVars[id].potential += fPot;
  
  return;
}

// initialize accelerations and potentials to zero
__global__ void ZeroVars(VariablePartData *particleVars, int nVars) {
    int id;
    id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(id >= nVars) return;

    particleVars[id].a.x = 0.0;
    particleVars[id].a.y = 0.0;
    particleVars[id].a.z = 0.0;
    particleVars[id].potential = 0.0;
    particleVars[id].dtGrav = 0.0;
}

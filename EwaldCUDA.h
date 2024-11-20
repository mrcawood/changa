#ifndef _EWALD_CUDA_H_
#define _EWALD_CUDA_H_ 

#include "HostCUDA.h"

#define NEWH 80
#define BLOCK_SIZE 128

/** @brief Data for the Ewald h loop in the CUDA kernel
 */
typedef struct {
  cudatype hx, hy, hz; 
  cudatype hCfac, hSfac; 
} EwtData;

/** @brief CUDA version of complete MultipoleMoments for Ewald
 */
typedef struct {
#ifndef HEXADECAPOLE
  cudatype xx, xy, xz, yy, yz, zz; 
#endif
  cudatype totalMass; 
  cudatype cmx, cmy, cmz; 

} MultipoleMomentsData; 

/** @brief CUDA version of MOMC for Ewald
 */
typedef struct {
  cudatype m;
  cudatype xx,yy,xy,xz,yz;
  cudatype xxx,xyy,xxy,yyy,xxz,yyz,xyz;
  cudatype xxxx,xyyy,xxxy,yyyy,xxxz,yyyz,xxyy,xxyz,xyyz;
  cudatype zz;
  cudatype xzz,yzz,zzz;
  cudatype xxzz,xyzz,xzzz,yyzz,yzzz,zzzz;
} MomcData;

/** @brief Parameters and data for Ewald in the CUDA kernel
 */
typedef struct {
  MultipoleMomentsData mm; 
  MomcData momcRoot;
  
  int n, nReps, nEwReps, nEwhLoop;
  cudatype L, fEwCut, alpha, alpha2, k1, ka, fEwCut2, fInner2;

} EwaldReadOnlyData; 

/// @brief structure to hold information specific to GPU Ewald
typedef struct {
  int EwaldRange[2];            /**< First and last particle on the
                                 * GPU; only used for small phase  */
  EwtData *ewt;                 /**< h-loop table  */
  EwaldReadOnlyData *cachedData; /**< Root moment and other Ewald parameters  */
} EwaldData; 

__global__ void EwaldKernel(CompactPartData *particleCores, VariablePartData *particleVars, int largephase, int First, int Last);

#endif


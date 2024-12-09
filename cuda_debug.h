#ifndef _CUDA_DEBUG_H_
#define _CUDA_DEBUG_H_

#include "charm++.h"

// Debug print control
#ifdef CUDA_DEBUG
#define CUDA_MEM_DEBUG(fmt, ...) CmiPrintf(fmt, ##__VA_ARGS__)
#else
#define CUDA_MEM_DEBUG(fmt, ...)
#endif

// Memory statistics print control - separate from debug output
#ifdef CUDA_MEMORY_STATS
#define CUDA_MEM_STATS(fmt, ...) CmiPrintf(fmt, ##__VA_ARGS__)
#else
#define CUDA_MEM_STATS(fmt, ...)
#endif

#endif // _CUDA_DEBUG_H_ 
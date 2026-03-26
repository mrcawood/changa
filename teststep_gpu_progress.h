#ifndef TESTSTEP_GPU_PROGRESS_H
#define TESTSTEP_GPU_PROGRESS_H

#if defined(CUDA) && defined(TESTSTEP_GPU_PROGRESS_DIAG)

#include <atomic>
#include <cstdint>

/// Async GPU / HAPI paths (submit → hapiAddCallback → Charm callback).
enum GpuProgPath : int {
  GP_DM_LOCAL_XFER = 0,   ///< DataManagerTransferLocalTree → startLocalWalk
  GP_DM_REMOTE_XFER = 1,  ///< DataManagerTransferRemoteChunk → resumeRemoteChunk
  GP_DM_LOCAL_TREE = 2,   ///< DataManagerLocalTreeWalk → finishLocalWalk
  GP_PELIST = 3,          ///< PEList *DataTransfer* → finishWalkCb
  GP_PARTVAR_BACK = 4,    ///< TransferParticleVarsBack → update path
  GP_EWALD = 5,           ///< DataManagerEwald → finishEwaldGPU
  GP_COUNT = 6
};

extern std::atomic<int> g_gpu_prog_big_step;

extern std::atomic<std::uint64_t> g_gpu_prog_submit[GP_COUNT];
extern std::atomic<std::uint64_t> g_gpu_prog_hapi_enqueue[GP_COUNT];
extern std::atomic<std::uint64_t> g_gpu_prog_cb_fired[GP_COUNT];
extern std::atomic<std::uint64_t> g_gpu_prog_cuda_err_events[GP_COUNT];
extern std::atomic<int> g_gpu_prog_last_cuda_err_code;

void gpu_prog_set_big_step(int big_step_display);

void gpu_prog_submit(GpuProgPath p);
void gpu_prog_hapi_enqueue(GpuProgPath p);
void gpu_prog_cb_fired(GpuProgPath p);
void gpu_prog_note_cuda_err(GpuProgPath p, int cuda_err_as_int);

/// One line per PE per big step: call from TreePiece::startGravity entry (first gravity in that step on this PE).
void gpu_prog_try_snapshot_first_gravity_this_bigstep();

/// One line per OS process per big step: after local H2D xfer callback work, before local GPU tree-walk enqueue.
void gpu_prog_try_snapshot_post_local_xfer();

/// Before `abort()` from CUDA fatal check (HostCUDA cudaErrorDie).
void gpu_prog_snapshot_fatal_cuda(int cuda_err, const char *code_expr, const char *file, int line);

#endif /* CUDA && TESTSTEP_GPU_PROGRESS_DIAG */

#endif /* TESTSTEP_GPU_PROGRESS_H */

#if defined(CUDA) && defined(TESTSTEP_GPU_PROGRESS_DIAG)

#include "teststep_gpu_progress.h"

#include "converse.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#include <mpi.h>
#define GPU_PROG_HAVE_MPI 1

#ifndef GPU_PROG_SNAPSHOT_MAX_PE
#define GPU_PROG_SNAPSHOT_MAX_PE 16384
#endif

std::atomic<int> g_gpu_prog_big_step{-1};

std::atomic<std::uint64_t> g_gpu_prog_submit[GP_COUNT];
std::atomic<std::uint64_t> g_gpu_prog_hapi_enqueue[GP_COUNT];
std::atomic<std::uint64_t> g_gpu_prog_cb_fired[GP_COUNT];
std::atomic<std::uint64_t> g_gpu_prog_cuda_err_events[GP_COUNT];
std::atomic<int> g_gpu_prog_last_cuda_err_code{0};

namespace {

std::atomic<int> s_last_first_grav_big[GPU_PROG_SNAPSHOT_MAX_PE];
std::atomic<int> s_last_post_local_xfer_big{-2147483640};

static int query_mpi_world_rank() {
#if GPU_PROG_HAVE_MPI
  int flag = 0;
  if (MPI_Initialized(&flag) != MPI_SUCCESS || !flag)
    return -1;
  int r = -1;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &r) != MPI_SUCCESS)
    return -1;
  return r;
#else
  return -1;
#endif
}

static void get_hostname_short(char *out, size_t cap) {
  if (cap == 0)
    return;
  out[0] = '\0';
  if (gethostname(out, cap) != 0)
    (void)std::snprintf(out, cap, "?");
  out[cap - 1] = '\0';
}

static void emit_snapshot(const char *tag, int sub_hint, int force_last_cuda) {
  const int mpi_r = query_mpi_world_rank();
  const int gpe = CmiMyPeGlobal();
  const int pid = static_cast<int>(getpid());
  char host[48];
  get_hostname_short(host, sizeof(host));

  const int part = CmiMyPartition();
  const int pe = CmiMyPe();
  const int node = CmiMyNode();
  const int gnode = CmiMyNodeGlobal();
  const int b = g_gpu_prog_big_step.load(std::memory_order_relaxed);
  const int last_cuda =
      force_last_cuda >= 0 ? force_last_cuda
                           : g_gpu_prog_last_cuda_err_code.load(std::memory_order_relaxed);
  unsigned pend_mask = 0;
  unsigned hs_skew = 0;
  std::int64_t out[GP_COUNT];
  std::uint64_t ts = 0, th = 0, tc = 0;
  for (int i = 0; i < GP_COUNT; i++) {
    const std::uint64_t s = g_gpu_prog_submit[i].load(std::memory_order_relaxed);
    const std::uint64_t h = g_gpu_prog_hapi_enqueue[i].load(std::memory_order_relaxed);
    const std::uint64_t c = g_gpu_prog_cb_fired[i].load(std::memory_order_relaxed);
    ts += s;
    th += h;
    tc += c;
    if (h > c)
      pend_mask |= (1u << i);
    if (s != h)
      hs_skew |= (1u << i);
    out[i] = static_cast<std::int64_t>(s) - static_cast<std::int64_t>(c);
  }
  std::fprintf(stderr,
               "TESTSTEP_GPU_PROGRESS tag=%s mpi_r=%d gpe=%d pid=%d host=%s part=%d pe=%d node=%d "
               "gnode=%d big=%d sub=%d pend_mask=0x%x hs_skew=0x%x out=%lld,%lld,%lld,%lld,%lld,%lld "
               "ts=%llu th=%llu tc=%llu last_cuda=%d\n",
               tag != nullptr ? tag : "?", mpi_r, gpe, pid, host, part, pe, node, gnode, b, sub_hint,
               pend_mask, hs_skew, (long long)out[0], (long long)out[1], (long long)out[2],
               (long long)out[3], (long long)out[4], (long long)out[5], (unsigned long long)ts,
               (unsigned long long)th, (unsigned long long)tc, last_cuda);
}

} // namespace

void gpu_prog_set_big_step(int big_step_display) {
  g_gpu_prog_big_step.store(big_step_display, std::memory_order_relaxed);
}

void gpu_prog_submit(GpuProgPath p) {
  if (p >= 0 && p < GP_COUNT)
    g_gpu_prog_submit[p]++;
}

void gpu_prog_hapi_enqueue(GpuProgPath p) {
  if (p >= 0 && p < GP_COUNT)
    g_gpu_prog_hapi_enqueue[p]++;
}

void gpu_prog_cb_fired(GpuProgPath p) {
  if (p >= 0 && p < GP_COUNT)
    g_gpu_prog_cb_fired[p]++;
}

void gpu_prog_note_cuda_err(GpuProgPath p, int cuda_err_as_int) {
  g_gpu_prog_last_cuda_err_code.store(cuda_err_as_int, std::memory_order_relaxed);
  if (p >= 0 && p < GP_COUNT)
    g_gpu_prog_cuda_err_events[p]++;
}

void gpu_prog_try_snapshot_first_gravity_this_bigstep() {
  const int pe = CmiMyPe();
  if (pe < 0 || pe >= GPU_PROG_SNAPSHOT_MAX_PE)
    return;
  const int b = g_gpu_prog_big_step.load(std::memory_order_relaxed);
  int prev = s_last_first_grav_big[pe].load(std::memory_order_relaxed);
  while (prev != b) {
    if (s_last_first_grav_big[pe].compare_exchange_weak(prev, b, std::memory_order_acq_rel,
                                                        std::memory_order_relaxed)) {
      emit_snapshot("first_gravity", -1, -1);
      return;
    }
  }
}

/// Once per OS process per big step: local H2D xfer has completed; TreePieces notified;
/// local GPU tree walk kernel not yet enqueued from this callback.
void gpu_prog_try_snapshot_post_local_xfer() {
  const int b = g_gpu_prog_big_step.load(std::memory_order_relaxed);
  int prev = s_last_post_local_xfer_big.load(std::memory_order_relaxed);
  while (prev != b) {
    if (s_last_post_local_xfer_big.compare_exchange_weak(prev, b, std::memory_order_acq_rel,
                                                         std::memory_order_relaxed)) {
      emit_snapshot("post_local_xfer", -1, -1);
      return;
    }
  }
}

void gpu_prog_snapshot_fatal_cuda(int cuda_err, const char *code_expr, const char *file, int line) {
  g_gpu_prog_last_cuda_err_code.store(cuda_err, std::memory_order_relaxed);
  const int mpi_r = query_mpi_world_rank();
  const int gpe = CmiMyPeGlobal();
  const int pid = static_cast<int>(getpid());
  char host[48];
  get_hostname_short(host, sizeof(host));
  std::fprintf(stderr,
               "TESTSTEP_GPU_PROGRESS tag=fatal_cuda mpi_r=%d gpe=%d pid=%d host=%s part=%d pe=%d "
               "node=%d gnode=%d big=%d cuda_err=%d expr=%s at %s:%d\n",
               mpi_r, gpe, pid, host, CmiMyPartition(), CmiMyPe(), CmiMyNode(), CmiMyNodeGlobal(),
               g_gpu_prog_big_step.load(std::memory_order_relaxed), cuda_err,
               code_expr != nullptr ? code_expr : "?", file != nullptr ? file : "?", line);
  emit_snapshot("fatal_cuda_pre_abort", -1, cuda_err);
}

#endif /* CUDA && TESTSTEP_GPU_PROGRESS_DIAG */

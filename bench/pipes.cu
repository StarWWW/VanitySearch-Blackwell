// Microbenchmark: per-SM throughput of the integer instructions the kernel is made of.
// Build: nvcc -O3 -arch=sm_120 -o pipes.exe pipes.cu      Run: pipes.exe
// Each test runs 8 independent dependency chains per thread so latency is hidden;
// the result is warp-instructions per SM per clock (4.0 = one per scheduler per clock).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define ITER 4096
#define CH 8

#define OP_SHF(x)   asm volatile("shf.r.wrap.b32 %0, %0, %0, 7;" : "+r"(x))
#define OP_LOP3(x)  asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(s1), "r"(s2))
#define OP_IADD3(x) asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(s1))
#define OP_ADDC(x)  asm volatile("add.cc.u32 %0, %0, %1;\n\taddc.u32 %0, %0, %2;" : "+r"(x) : "r"(s1), "r"(s2))
#define OP_IMAD(x)  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(s1), "r"(s2))
#define OP_PRMT(x)  asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(x) : "r"(s1))
#define OP_LEA(x)   asm volatile("{.reg .u32 t; shl.b32 t, %0, 5; add.u32 %0, t, %1;}" : "+r"(x) : "r"(s1))

#define OP_ADD64(y) asm volatile("add.u64 %0, %0, %1;" : "+l"(y) : "l"(d1))
#define OP_WIDE(y)  asm volatile("{.reg .u32 lo, hi; mov.b64 {lo, hi}, %0; mad.wide.u32 %0, lo, %1, %0;}" : "+l"(y) : "r"(s1))

#define DEF32(name, OP) \
__global__ void k_##name(uint32_t *out, uint32_t s1, uint32_t s2) { \
  uint32_t x[CH]; for (int c = 0; c < CH; c++) x[c] = threadIdx.x + c; \
  for (int i = 0; i < ITER; i++) { _Pragma("unroll") for (int c = 0; c < CH; c++) OP(x[c]); } \
  uint32_t r = 0; for (int c = 0; c < CH; c++) r ^= x[c]; if (r == 0x12345678) out[0] = r; }

#define DEF64(name, OP) \
__global__ void k_##name(uint32_t *out, uint32_t s1, uint32_t s2) { \
  uint64_t d1 = ((uint64_t)s2 << 32) | s1; uint64_t x[CH]; for (int c = 0; c < CH; c++) x[c] = threadIdx.x + c; \
  for (int i = 0; i < ITER; i++) { _Pragma("unroll") for (int c = 0; c < CH; c++) OP(x[c]); } \
  uint64_t r = 0; for (int c = 0; c < CH; c++) r ^= x[c]; if (r == 0x12345678) out[0] = (uint32_t)r; }

// two different ops interleaved: do they issue to different pipes in parallel?
#define DEFMIX(name, OPA, OPB) \
__global__ void k_##name(uint32_t *out, uint32_t s1, uint32_t s2) { \
  uint32_t x[CH]; for (int c = 0; c < CH; c++) x[c] = threadIdx.x + c; \
  for (int i = 0; i < ITER; i++) { _Pragma("unroll") for (int c = 0; c < CH; c += 2) { OPA(x[c]); OPB(x[c + 1]); } } \
  uint32_t r = 0; for (int c = 0; c < CH; c++) r ^= x[c]; if (r == 0x12345678) out[0] = r; }

DEF32(shf, OP_SHF)
DEF32(lop3, OP_LOP3)
DEF32(iadd3, OP_IADD3)
DEF32(addc, OP_ADDC)
DEF32(imad, OP_IMAD)
DEF32(prmt, OP_PRMT)
DEF32(lea, OP_LEA)
DEF64(add64, OP_ADD64)
DEF64(wide, OP_WIDE)
DEFMIX(shf_imad, OP_SHF, OP_IMAD)
DEFMIX(shf_lop3, OP_SHF, OP_LOP3)
DEFMIX(shf_iadd3, OP_SHF, OP_IADD3)
DEFMIX(lop3_imad, OP_LOP3, OP_IMAD)
DEFMIX(iadd3_imad, OP_IADD3, OP_IMAD)
DEFMIX(shf_prmt, OP_SHF, OP_PRMT)

typedef void (*kfn)(uint32_t *, uint32_t, uint32_t);

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  uint32_t *out; cudaMalloc(&out, 4);
  struct { const char *n; kfn f; int insPerOp; } t[] = {
    {"SHF", k_shf, 1}, {"LOP3", k_lop3, 1}, {"IADD3", k_iadd3, 1}, {"IADD+IADD.X (carry pair)", k_addc, 2},
    {"IMAD", k_imad, 1}, {"PRMT", k_prmt, 1}, {"LEA (shl+add)", k_lea, 1}, {"IADD.64", k_add64, 1},
    {"IMAD.WIDE", k_wide, 1}, {"SHF | IMAD", k_shf_imad, 1}, {"SHF | LOP3", k_shf_lop3, 1},
    {"SHF | IADD3", k_shf_iadd3, 1}, {"LOP3 | IMAD", k_lop3_imad, 1}, {"IADD3 | IMAD", k_iadd3_imad, 1},
    {"SHF | PRMT", k_shf_prmt, 1}};
  int blocks = p.multiProcessorCount * 8, threads = 384;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  printf("%s, %d SMs, nominal clock %d MHz (results scale with the real clock)\n", p.name, p.multiProcessorCount, clk / 1000);
  for (auto &x : t) {
    x.f<<<blocks, threads>>>(out, 3, 5); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 5; r++) x.f<<<blocks, threads>>>(out, 3, 5);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    double warpIns = 5.0 * blocks * (threads / 32) * (double)ITER * CH * x.insPerOp;
    double perSmClk = warpIns / (ms * 1e-3) / p.multiProcessorCount / (clk * 1e3);
    printf("  %-26s %5.2f warp-inst/SM/clk (nominal clk)\n", x.n, perSmClk);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(e));
  return 0;
}

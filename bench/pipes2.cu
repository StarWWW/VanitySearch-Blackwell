// Which integer instructions share the ALU pipe with SHF on sm_120?
// Each "A | B" test runs two independent chain families interleaved; if A and B use
// different pipes the combined rate is ~ rate(A) + rate(B), if they share one it stays ~ rate(A).
// Build: nvcc -O3 -arch=sm_120 -o pipes2.exe pipes2.cu
// Read the result with Nsight Compute pipe counters (sm__inst_executed_pipe_*), e.g.
//   ncu --kernel-id ::regex:k_:1 --metrics sm__inst_executed_pipe_aluheavy.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed.sum pipes2.exe
// The printed timing ratios are latency-bound and not meaningful on their own.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define ITER 2048
#define CH 8

__device__ __constant__ uint32_t c_one = 1;

// x: chain value, y: partner chain value (keeps the compiler from folding), s: kernel scalar
#define SHF_(x, y)   asm volatile("shf.r.wrap.b32 %0, %0, %0, 7;" : "+r"(x))
#define LOP3_(x, y)  asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(y), "r"(s))
#define IADD3_(x, y) asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(x) : "r"(y), "r"(s))
#define IADD2_(x, y) asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(y))
#define IMAD1_(x, y) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(one), "r"(y))
#define LEA_(x, y)   asm volatile("{.reg .u32 t; shl.b32 t, %0, 3; add.u32 %0, t, %1;}" : "+r"(x) : "r"(y))
#define ADDC_(x, y)  asm volatile("add.cc.u32 %0, %0, %1; addc.u32 %0, %0, %2;" : "+r"(x) : "r"(y), "r"(s))
#define MOVSEL_(x, y) asm volatile("{.reg .pred p; setp.lt.u32 p, %0, %1; selp.b32 %0, %1, %0, p;}" : "+r"(x) : "r"(y))

#define DEF1(name, OP, NI) \
__global__ void k_##name(uint32_t *out, uint32_t s) { \
  uint32_t one = c_one; uint32_t x[CH]; for (int c = 0; c < CH; c++) x[c] = threadIdx.x * 7 + c; \
  for (int i = 0; i < ITER; i++) { _Pragma("unroll") for (int c = 0; c < CH; c++) OP(x[c], x[c ^ 1]); } \
  uint32_t r = 0; for (int c = 0; c < CH; c++) r ^= x[c]; if (r == 0x12345678) out[0] = r + one; }

#define DEF2(name, OPA, OPB) \
__global__ void k_##name(uint32_t *out, uint32_t s) { \
  uint32_t one = c_one; uint32_t x[CH]; for (int c = 0; c < CH; c++) x[c] = threadIdx.x * 7 + c; \
  for (int i = 0; i < ITER; i++) { _Pragma("unroll") for (int c = 0; c < CH; c += 4) { \
    OPA(x[c], x[c + 1]); OPB(x[c + 2], x[c + 3]); OPA(x[c + 1], x[c]); OPB(x[c + 3], x[c + 2]); } } \
  uint32_t r = 0; for (int c = 0; c < CH; c++) r ^= x[c]; if (r == 0x12345678) out[0] = r + one; }

DEF1(shf, SHF_, 1) DEF1(lop3, LOP3_, 1) DEF1(iadd3, IADD3_, 1) DEF1(iadd2, IADD2_, 1) DEF1(imad1, IMAD1_, 1)
DEF1(lea, LEA_, 1) DEF1(addc, ADDC_, 2) DEF1(sel, MOVSEL_, 2)
DEF2(shf_iadd3, SHF_, IADD3_) DEF2(shf_iadd2, SHF_, IADD2_) DEF2(shf_imad1, SHF_, IMAD1_) DEF2(shf_lea, SHF_, LEA_)
DEF2(shf_addc, SHF_, ADDC_) DEF2(shf_lop3, SHF_, LOP3_) DEF2(lop3_iadd3, LOP3_, IADD3_) DEF2(lop3_imad1, LOP3_, IMAD1_)
DEF2(iadd3_imad1, IADD3_, IMAD1_) DEF2(shf_sel, SHF_, MOVSEL_)

typedef void (*kfn)(uint32_t *, uint32_t);

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  uint32_t *out; cudaMalloc(&out, 4);
  // instructions per OP invocation (A and B averaged for mixes)
  struct { const char *n; kfn f; double ins; } t[] = {
    {"SHF", k_shf, 1}, {"LOP3", k_lop3, 1}, {"IADD3", k_iadd3, 1}, {"IADD (2-in)", k_iadd2, 1},
    {"IMAD x*one+y", k_imad1, 1}, {"LEA", k_lea, 1}, {"IADD.CC+IADD.X", k_addc, 2}, {"ISETP+SEL", k_sel, 2},
    {"SHF | IADD3", k_shf_iadd3, 1}, {"SHF | IADD", k_shf_iadd2, 1}, {"SHF | IMAD", k_shf_imad1, 1},
    {"SHF | LEA", k_shf_lea, 1}, {"SHF | IADD.CC+X", k_shf_addc, 1.5}, {"SHF | LOP3", k_shf_lop3, 1},
    {"LOP3 | IADD3", k_lop3_iadd3, 1}, {"LOP3 | IMAD", k_lop3_imad1, 1}, {"IADD3 | IMAD", k_iadd3_imad1, 1},
    {"SHF | ISETP+SEL", k_shf_sel, 1.5}};
  int blocks = p.multiProcessorCount * 8, threads = 384;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  // measure the real clock-independent ratio against SHF
  double shf = 0;
  printf("%s, %d SMs. Rates relative to SHF alone (SHF = 1.00):\n", p.name, p.multiProcessorCount);
  for (auto &x : t) {
    x.f<<<blocks, threads>>>(out, 3); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 5; r++) x.f<<<blocks, threads>>>(out, 3);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    double rate = 5.0 * blocks * threads * (double)ITER * CH * x.ins / (ms * 1e-3);
    if (shf == 0) shf = rate;
    printf("  %-18s %5.2f\n", x.n, rate / shf);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(e));
  return 0;
}

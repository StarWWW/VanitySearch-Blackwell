// Field multiplication microbenchmark: VanitySearch's 64-bit-limb _ModMult vs a 32-bit-limb
// row-scanning 256x256 product (PTX carry chains) feeding the same 512->256 reduction.
// Build: nvcc -O3 -arch=sm_120 -I.. -o fieldmul.exe fieldmul.cu   Run: fieldmul.exe > out.txt
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../GPU/GPUMath.h"

#define MULLO32(d, a, b)       asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b))
#define MADLO_CC(d, a, b, c)   asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c))
#define MADCLO_CC(d, a, b, c)  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c))
#define MADHI_CC(d, a, b, c)   asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c))
#define MADCHI_CC(d, a, b, c)  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c))
#define MADCHI(d, a, b, c)     asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c))
#define ADDC0(d)               asm volatile("addc.u32 %0, 0, 0;" : "=r"(d))

// 256x256 -> 512 with 32-bit limbs. Row i adds a*b[i] at limb i: a lo pass (carry chain over
// the low halves) and a hi pass (carry chain over the high halves, one limb up). After row i the
// partial product is < 2^(256+32(i+1)) so it fits in limbs 0..i+8 and the hi pass never carries out.
__device__ __forceinline__ void Mul256_32(uint64_t *r512, const uint64_t *a64, const uint64_t *b64) {
  uint32_t a[8], b[8], t[16];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    a[2 * i] = (uint32_t)a64[i]; a[2 * i + 1] = (uint32_t)(a64[i] >> 32);
    b[2 * i] = (uint32_t)b64[i]; b[2 * i + 1] = (uint32_t)(b64[i] >> 32);
  }
#pragma unroll
  for (int j = 0; j < 8; j++) MULLO32(t[j], a[j], b[0]);
  MADHI_CC(t[1], a[0], b[0], t[1]);
#pragma unroll
  for (int j = 1; j < 7; j++) MADCHI_CC(t[j + 1], a[j], b[0], t[j + 1]);
  MADCHI(t[8], a[7], b[0], 0);
#pragma unroll
  for (int i = 1; i < 8; i++) {
    MADLO_CC(t[i], a[0], b[i], t[i]);
#pragma unroll
    for (int j = 1; j < 8; j++) MADCLO_CC(t[i + j], a[j], b[i], t[i + j]);
    ADDC0(t[i + 8]);
    MADHI_CC(t[i + 1], a[0], b[i], t[i + 1]);
#pragma unroll
    for (int j = 1; j < 7; j++) MADCHI_CC(t[i + j + 1], a[j], b[i], t[i + j + 1]);
    MADCHI(t[i + 8], a[7], b[i], t[i + 8]);
  }
#pragma unroll
  for (int k = 0; k < 8; k++) r512[k] = (uint64_t)t[2 * k] | ((uint64_t)t[2 * k + 1] << 32);
}

// Same 512->256 reduction as _ModMult
__device__ __forceinline__ void Reduce512(uint64_t *r, uint64_t *r512) {
  uint64_t t[NBBLOCK];
  uint64_t ah, al;
  ReduceMult(t, (r512 + 4));
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);
  UADD1(t[4], 0ULL);
  UMULLO(al, t[4], 0x1000003D1ULL);
  UMULHI(ah, t[4], 0x1000003D1ULL);
  UADDO(r[0], r512[0], al);
  UADDC(r[1], r512[1], ah);
  UADDC(r[2], r512[2], 0ULL);
  UADD(r[3], r512[3], 0ULL);
}

__device__ __forceinline__ void ModMult32(uint64_t *r, uint64_t *a, uint64_t *b) {
  uint64_t r512[8];
  Mul256_32(r512, a, b);
  Reduce512(r, r512);
}

__global__ void check(const uint64_t *in, uint64_t *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t a[4], b[4], r1[4], r2[4];
  for (int k = 0; k < 4; k++) { a[k] = in[8 * i + k]; b[k] = in[8 * i + 4 + k]; }
  _ModMult(r1, a, b);
  ModMult32(r2, a, b);
  for (int k = 0; k < 4; k++) { out[8 * i + k] = r1[k]; out[8 * i + 4 + k] = r2[k]; }
}

#define ITER 2048
template <int V>
__global__ void speed(uint64_t *out, uint64_t seed) {
  uint64_t x[4][4], y[4] = {seed, seed * 3, seed * 5, seed * 7 & 0x7FFFFFFFFFFFFFFFULL};
  for (int c = 0; c < 4; c++) for (int k = 0; k < 4; k++) x[c][k] = (seed + threadIdx.x) * (k + c + 1);
  for (int i = 0; i < ITER; i++) {
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (V == 0) _ModMult(x[c], x[c], y); else ModMult32(x[c], x[c], y);
    }
  }
  uint64_t s = 0;
  for (int c = 0; c < 4; c++) for (int k = 0; k < 4; k++) s ^= x[c][k];
  if (s == 0x1234) out[0] = s;
}

static uint64_t rnd() { return ((uint64_t)rand() << 62) ^ ((uint64_t)rand() << 31) ^ (uint64_t)rand() ^ ((uint64_t)rand() << 47); }

int main() {
  const int n = 1 << 16;
  uint64_t *h = (uint64_t *)malloc(n * 64), *o = (uint64_t *)malloc(n * 64), *din, *dout;
  srand(12345);
  for (int i = 0; i < n * 8; i++) h[i] = rnd();
  // edge cases: values near p and near 2^256
  for (int k = 0; k < 4; k++) { h[k] = ~0ULL; h[4 + k] = ~0ULL; }
  h[8] = 0xFFFFFFFEFFFFFC2EULL; h[9] = h[10] = h[11] = ~0ULL; h[12] = 0xFFFFFFFEFFFFFC2EULL; h[13] = h[14] = h[15] = ~0ULL;
  cudaMalloc(&din, n * 64); cudaMalloc(&dout, n * 64);
  cudaMemcpy(din, h, n * 64, cudaMemcpyHostToDevice);
  check<<<n / 256, 256>>>(din, dout, n);
  cudaMemcpy(o, dout, n * 64, cudaMemcpyDeviceToHost);
  int diff = 0;
  for (int i = 0; i < n; i++) for (int k = 0; k < 4; k++) diff += o[8 * i + k] != o[8 * i + 4 + k];
  fprintf(stderr, "vectors %d, limbs differing between old and new: %d\n", n, diff);
  for (int i = 0; i < 64; i++)   // a, b, r_new for the Python cross-check
    printf("%016llx%016llx%016llx%016llx %016llx%016llx%016llx%016llx %016llx%016llx%016llx%016llx\n",
           h[8*i+3], h[8*i+2], h[8*i+1], h[8*i], h[8*i+7], h[8*i+6], h[8*i+5], h[8*i+4],
           o[8*i+7], o[8*i+6], o[8*i+5], o[8*i+4]);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int blocks = p.multiProcessorCount * 16, threads = 256;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int v = 0; v < 2; v++) for (int rep = 0; rep < 2; rep++) {
    if (v == 0) speed<0><<<blocks, threads>>>(dout, 7); else speed<1><<<blocks, threads>>>(dout, 7);
    cudaEventRecord(e0);
    for (int r = 0; r < 3; r++) { if (v == 0) speed<0><<<blocks, threads>>>(dout, 7); else speed<1><<<blocks, threads>>>(dout, 7); }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    if (rep) fprintf(stderr, "%s: %.2f Gmul/s\n", v ? "32-bit rows " : "VanitySearch", 3.0 * blocks * threads * ITER * 4 / (ms * 1e6));
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
  return 0;
}

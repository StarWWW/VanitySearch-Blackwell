/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Modified in 2026 for NVIDIA Blackwell (sm_120) throughput, see OPTIMIZATION.md and the git history.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"
#include "../Vanity.h"

#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUCompute.h"
#include <iostream>

int _ConvertSMVer2Cores(int major, int minor) {

    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86,  128},
        {0x87,  128},
        {0x89,  128},
        {0x90,  128},
        {0xa0,  128},
        {0xa1,  128},
        {0xa3,  128},
        {0xb0,  128},
        {0xc0,  128},
        {0xc1,  128},
        {-1, -1} };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    return 0;

}



#define GRP_SIZE VS_GRP
#define STEP_SIZE GRP_SIZE*1

// Launch configuration, overridable at build time (-DVS_TPB=... etc.)
// 128 threads/block (barriers then sync 4 warps instead of 8) with 2x the blocks keeps the
// threads per launch unchanged; -2.1% cycles per key on sm_120 (bench/cycles.py).
#ifndef VS_TPB
#define VS_TPB 128
#endif
#ifndef VS_BPSM
#define VS_BPSM 256
#endif
#ifdef VS_MINBLOCKS
#define VS_BOUNDS __launch_bounds__(VS_TPB, VS_MINBLOCKS)
#else
#define VS_BOUNDS
#endif

// Block-wide barriers in comp_keys keep the warps of a block on the same Gx[i]/Gy[i] constants.
// No shared memory is used, so they are not needed for correctness (VS_NOBARRIER=1 drops them).
#if defined(VS_NOBARRIER) && VS_NOBARRIER
#define VS_SYNC()
#else
#define VS_SYNC() __syncthreads()
#endif

__global__ void VS_BOUNDS comp_keys(address_t* sAddress, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {


    uint64_t* startx = keys + (blockIdx.x * blockDim.x) * 8;
    uint64_t* starty = keys + (blockIdx.x * blockDim.x) * 8 + 4 * blockDim.x;


    uint64_t dx[4];  
    uint64_t px[4];
    uint64_t py[4];
    uint64_t dy[4];
    uint64_t sxn[4];
    uint64_t syn[4];
    uint64_t sx[4];
    uint64_t sy[4];
    uint64_t sx_gx[4];
    uint8_t odd_py;
    uint32_t h[5];
    uint64_t inverse[5];

    uint64_t subp[GRP_SIZE/2][4];
    

    VS_SYNC();
    Load256A(sx, startx);
    Load256A(sy, starty);


    uint32_t i;

    // Check starting point
    odd_py = sy[0] & 1;
    _GetHash160Comp(sx, odd_py, (uint8_t*)h);
    CheckPoint(h, GRP_SIZE / 2, sAddress, lookup32, out);
    VS_SYNC();

    ModSub256(sxn, _2Gnx, sx);
    Load256(subp[GRP_SIZE / 2 - 1], sxn);
    for (i = GRP_SIZE / 2 - 1; i > 0; i--) {
        ModSub256(syn, Gx[i], sx);
        _ModMult(sxn, syn);
        Load256(subp[i - 1], sxn);
    }

    ModSub256(inverse, Gx[0], sx);
    _ModMult(inverse, sxn);


    inverse[4] = 0;
    _ModInv(inverse);

    VS_SYNC();
    
    ModNeg256(syn, sy);
    ModNeg256(sxn, sx);

    for (i = 0; i < GRP_SIZE / 2 - 1; i++) {

        VS_SYNC();
        ModSub256(sx_gx, Gx[i], sxn);

        _ModMult(dx, subp[i], inverse);

        //////////////////

        ModSub256(dy, Gy[i], sy);
        _ModMult(dy, dx);
        _ModSqr(px, dy);
        ModSub256(px, sx_gx);

        ModSub256(py, sx, px);
        _ModMult(py, dy);
        ModSub256isOdd(py, sy, &odd_py);

        _GetHash160Comp(px, odd_py, (uint8_t*)h);
        CheckPoint(h, GRP_SIZE / 2 + (i + 1), sAddress, lookup32, out);

        //////////////////

        VS_SYNC();

        ModSub256(dy, syn, Gy[i]);
        _ModMult(dy, dx);
        _ModSqr(px, dy);
        ModSub256(px, sx_gx);

        ModSub256(py, px, sx);
        _ModMult(py, dy);
        ModSub256isOdd(syn, py, &odd_py);

        _GetHash160Comp(px, odd_py, (uint8_t*)h);
        CheckPoint(h, GRP_SIZE / 2 - (i + 1), sAddress, lookup32, out);

        //////////////////

        ModSub256(dx, Gx[i], sx);
        _ModMult(inverse, dx);

    }

    VS_SYNC();

    _ModMult(dx, subp[i], inverse);

    ModSub256(dy, syn, Gy[i]);
    _ModMult(dy, dx);
    _ModSqr(px, dy);
    ModSub256(px, sx);
    ModSub256(px, Gx[i]);

    ModSub256(py, px, sx);
    _ModMult(py, dy);
    ModSub256isOdd(syn, py, &odd_py);

    _GetHash160Comp(px, odd_py, (uint8_t*)h);
    CheckPoint(h, 0, sAddress, lookup32, out);

    //////////////////

    VS_SYNC();

    ModSub256(dy, _2Gny, sy);
    ModSub256(dx, Gx[i], sx);
    _ModMult(inverse, dx);

    _ModMult(dy, inverse);
    _ModSqr(px, dy);
    ModSub256(px, sx);
    ModSub256(px, _2Gnx);

    ModSub256(py, _2Gnx, px);
    _ModMult(py, dy);
    ModSub256(py, _2Gny);               

    VS_SYNC();
    Store256A(startx, px);
    Store256A(starty, py);


}


// ---------------------------------------------------------------------------------------
int NB_THREAD_PER_GROUP;

using namespace std;



GPUEngine::GPUEngine(int gpuId, uint32_t maxFound) {

    // Every buffer starts out unallocated so that the destructor is safe after a failed init
    inputAddress = NULL;
    inputAddressPinned = NULL;
    inputAddressLookUp = NULL;
    inputAddressLookUpPinned = NULL;
    bloomBits = NULL;
    inputKey = NULL;
    inputKeyPinned = NULL;
    outputBuffer = NULL;
    outputBufferPinned = NULL;
    initialised = false;
    lostWarning = false;
    nbThread = 0;
    this->maxFound = maxFound;
    outputSize = 0;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        fprintf(stderr, "GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
        return;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        fprintf(stderr, "GPUEngine: There are no available device(s) that support CUDA\n");
        return;
    }

    if (gpuId < 0 || gpuId >= deviceCount) {
        fprintf(stderr, "GPUEngine: invalid GPU id %d (%d device(s) found, use -l to list them)\n", gpuId, deviceCount);
        return;
    }

    cudaDeviceProp deviceProp;
    error_id = cudaGetDeviceProperties(&deviceProp, gpuId);
    if (error_id != cudaSuccess) {
        fprintf(stderr, "GPUEngine: cudaGetDeviceProperties %s\n", cudaGetErrorString(error_id));
        return;
    }

    NB_THREAD_PER_GROUP = VS_TPB;                                          //////////////////  GRID SIZE ////////////////
    int nbThreadGroup = deviceProp.multiProcessorCount * VS_BPSM;

    if (!randomMode) {
        uint64_t powerOfTwo = 1;
        while (powerOfTwo <= nbThreadGroup) {  //  GET THE CLOSEST POWER OF 2
            powerOfTwo <<= 1;
        }

        powerOfTwo >>= 1;
        nbThreadGroup = powerOfTwo;
    }


    // Initialise CUDA
    cudaError_t err;

    err = cudaSetDevice(gpuId);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

   

    this->nbThread = nbThreadGroup * NB_THREAD_PER_GROUP;//////////////////////////////////////////////////////////////////
    this->maxFound = maxFound;
    this->outputSize = (maxFound * ITEM_SIZE + 4);

    char tmp[512];
    sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
    gpuId,deviceProp.name,deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
    nbThread / NB_THREAD_PER_GROUP,
    NB_THREAD_PER_GROUP);

    deviceName = std::string(tmp);

    // Prefer L1 (We do not use __shared__ at all)
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) {
        printf("GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

    //size_t stackSize = 49152;
    //err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    //if (err != cudaSuccess) {
    //  printf("GPUEngine: %s\n", cudaGetErrorString(err));
    //  return;
    //}

    /*
    size_t heapSize = ;
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      exit(0);
    }

    size_t size;
    cudaDeviceGetLimit(&size, cudaLimitStackSize);
    printf("Stack Size %lld\n", size);
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    printf("Heap Size %lld\n", size);
    */

    // Allocate memory
    err = cudaMalloc((void**)&inputAddress, _64K * 2);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate address memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&inputAddressPinned, _64K * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate address pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&inputKey, nbThread * 32 * 2);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&outputBuffer, outputSize);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&outputBufferPinned, outputSize, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }

    // The kernel drops hits beyond the output buffer instead of writing past its end
    err = cudaMemcpyToSymbol(_maxFound, &maxFound, sizeof(maxFound));
    if (err != cudaSuccess) {
        printf("GPUEngine: Set maxFound: %s\n", cudaGetErrorString(err));
        return;
    }

    searchMode = SEARCH_COMPRESSED;
    searchType = P2PKH;
    initialised = true;

}

GPUEngine::~GPUEngine() {

    ReleaseBuffers();

}

void GPUEngine::ReleaseBuffers() {

    if (inputKey) cudaFree(inputKey);
    if (inputAddress) cudaFree(inputAddress);
    if (inputAddressLookUp) cudaFree(inputAddressLookUp);
    if (bloomBits) cudaFree(bloomBits);
    if (outputBuffer) cudaFree(outputBuffer);
    if (inputAddressPinned) cudaFreeHost(inputAddressPinned);
    if (inputAddressLookUpPinned) cudaFreeHost(inputAddressLookUpPinned);
    if (inputKeyPinned) cudaFreeHost(inputKeyPinned);
    if (outputBufferPinned) cudaFreeHost(outputBufferPinned);

    inputKey = NULL;
    inputAddress = NULL;
    inputAddressLookUp = NULL;
    bloomBits = NULL;
    outputBuffer = NULL;
    inputAddressPinned = NULL;
    inputAddressLookUpPinned = NULL;
    inputKeyPinned = NULL;
    outputBufferPinned = NULL;

}

bool GPUEngine::IsInitialised() {
    return initialised;
}

void GPUEngine::PrintCudaInfo() {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        fprintf(stderr, "GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
        return;
    }

    if (deviceCount == 0) {
        printf("No CUDA capable device found\n");
        return;
    }

    for (int i = 0;i < deviceCount;i++) {

        cudaDeviceProp deviceProp;
        if (cudaGetDeviceProperties(&deviceProp, i) != cudaSuccess)
            continue;

        printf("GPU #%d %s (sm_%d%d, %d SMs, %.1f GB)\n", i, deviceProp.name, deviceProp.major, deviceProp.minor,
            deviceProp.multiProcessorCount, (double)deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    }

}



int GPUEngine::GetNbThread() {
    return nbThread;
}

void GPUEngine::SetSearchMode(int searchMode) {
    this->searchMode = searchMode;
}

void GPUEngine::SetSearchType(int searchType) {
    this->searchType = searchType;
}





bool GPUEngine::SetAddress(std::vector<address_t> addresses) {

    memset(inputAddressPinned, 0, _64K * 2);
    for (int i = 0;i < (int)addresses.size();i++)
        inputAddressPinned[addresses[i]] = 1;

    // Fill device memory
    cudaMemcpy(inputAddress, inputAddressPinned, _64K * 2, cudaMemcpyHostToDevice);

    // We do not need the input pinned memory anymore
    cudaFreeHost(inputAddressPinned);
    inputAddressPinned = NULL;
    lostWarning = false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetAddress: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;

}

bool GPUEngine::SetAddress(std::vector<LADDRESS> addresses, uint32_t totalAddress) {

    // Allocate memory for the second level of lookup tables
    cudaError_t err = cudaMalloc((void**)&inputAddressLookUp, (_64K + totalAddress) * 4);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate address lookup memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaHostAlloc(&inputAddressLookUpPinned, (_64K + totalAddress) * 4, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate address lookup pinned memory: %s\n", cudaGetErrorString(err));
        return false;
    }

    uint32_t offset = _64K;
    memset(inputAddressPinned, 0, _64K * 2);
    memset(inputAddressLookUpPinned, 0, _64K * 4);
    for (int i = 0; i < (int)addresses.size(); i++) {
        int nbLAddress = (int)addresses[i].lAddresses.size();
        inputAddressPinned[addresses[i].sAddress] = (uint16_t)nbLAddress;
        inputAddressLookUpPinned[addresses[i].sAddress] = offset;
        for (int j = 0; j < nbLAddress; j++) {
            inputAddressLookUpPinned[offset++] = addresses[i].lAddresses[j];
        }
    }

    if (offset != (_64K + totalAddress)) {
        printf("GPUEngine: Wrong totalAddress %d!=%d!\n", offset - _64K, totalAddress);
        return false;
    }

    // Fill device memory
    cudaMemcpy(inputAddress, inputAddressPinned, _64K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(inputAddressLookUp, inputAddressLookUpPinned, (_64K + totalAddress) * 4, cudaMemcpyHostToDevice);


    // We do not need the input pinned memory anymore
    cudaFreeHost(inputAddressPinned);
    inputAddressPinned = NULL;
    cudaFreeHost(inputAddressLookUpPinned);
    inputAddressLookUpPinned = NULL;
    lostWarning = false;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetAddress (large): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;

}

// Multi full-address search. The filter has 32..64 bits per address (rounded up to a power of
// two) and 8 probes: about 1.2 memory reads per key and a false positive rate of 4e-8..6e-6,
// which the CPU check removes. The probes are taken from the hash160 itself, which is uniform.
bool GPUEngine::SetBloom(const std::vector<const uint8_t*>& hash160s) {

    uint64_t n = hash160s.size();
    uint64_t nbBits = 1ULL << 18;                       // 32 KB minimum
    while (nbBits < n * 32 && nbBits < (1ULL << 32))
        nbBits <<= 1;
    uint32_t k = 8;
    size_t nbWords = (size_t)(nbBits / 32);

    std::vector<uint32_t> bits(nbWords, 0);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t h[5];
        memcpy(h, hash160s[i], 20);
        uint32_t bit = h[0];
        uint32_t step = h[1] | 1;
        for (uint32_t j = 0; j < k; j++) {
            uint32_t idx = bit & (uint32_t)(nbBits - 1);
            bits[idx >> 5] |= 1u << (idx & 31);
            bit += step;
        }
    }

    size_t setBits = 0;
    for (size_t i = 0; i < nbWords; i++) {
        uint32_t w = bits[i];
        while (w) { w &= w - 1; setBits++; }
    }
    double fill = (double)setBits / (double)nbBits;
    double fpRate = pow(fill, (double)k);

    cudaError_t err = cudaMalloc((void**)&bloomBits, nbWords * 4);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate bloom filter (%.1f MB): %s\n", nbWords * 4 / 1048576.0, cudaGetErrorString(err));
        return false;
    }
    cudaMemcpy(bloomBits, bits.data(), nbWords * 4, cudaMemcpyHostToDevice);

    uint32_t mask = (uint32_t)(nbBits - 1);
    const uint32_t* ptr = bloomBits;
    cudaMemcpyToSymbol(_bloomMask, &mask, sizeof(mask));
    cudaMemcpyToSymbol(_bloomBits, &ptr, sizeof(ptr));
    cudaMemcpyToSymbol(_bloomK, &k, sizeof(k));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetBloom: %s\n", cudaGetErrorString(err));
        return false;
    }

    double perLaunch = fpRate * (double)nbThread * (double)STEP_SIZE;
    printf("Bloom filter: %llu addresses, %.1f MB, %u probes, false positive rate %.1e (~%.0f per launch)\n",
        (unsigned long long)n, nbWords * 4 / 1048576.0, k, fpRate, perLaunch);
    if (perLaunch > maxFound / 4.0)
        printf("Warning: false positives may fill the hit buffer, increase -m (now %u)\n", maxFound);
    return true;

}

bool GPUEngine::SetSingleTarget(const uint8_t* hash160) {

    uint32_t h[5];
    uint32_t one = 1;
    memcpy(h, hash160, 20);
    cudaMemcpyToSymbol(_targetH160, h, sizeof(h));
    cudaMemcpyToSymbol(_singleTarget, &one, sizeof(one));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetSingleTarget: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;

}

int GPUEngine::GetStepSize() {

    return STEP_SIZE;

}

int GPUEngine::GetGroupSize() {

    return GRP_SIZE;

}


bool GPUEngine::callKernel() {

   
    // Reset nbFound
    cudaMemset(outputBuffer, 0, 4);

    comp_keys << < nbThread / NB_THREAD_PER_GROUP, NB_THREAD_PER_GROUP >> >
        (inputAddress, inputAddressLookUp, inputKey, outputBuffer);




    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
        return false;
    }

    //cudaFree(d_dx);


    return true;

}


bool GPUEngine::SetKeys(Point* p) {

    // Sets the starting keys for each thread
    // p must contains nbThread public keys

    for (int i = 0; i < nbThread; i += NB_THREAD_PER_GROUP) {
        for (int j = 0; j < NB_THREAD_PER_GROUP; j++) {

            inputKeyPinned[8 * i + j + 0 * NB_THREAD_PER_GROUP] = p[i + j].x.bits64[0];
            inputKeyPinned[8 * i + j + 1 * NB_THREAD_PER_GROUP] = p[i + j].x.bits64[1];
            inputKeyPinned[8 * i + j + 2 * NB_THREAD_PER_GROUP] = p[i + j].x.bits64[2];
            inputKeyPinned[8 * i + j + 3 * NB_THREAD_PER_GROUP] = p[i + j].x.bits64[3];

            inputKeyPinned[8 * i + j + 4 * NB_THREAD_PER_GROUP] = p[i + j].y.bits64[0];
            inputKeyPinned[8 * i + j + 5 * NB_THREAD_PER_GROUP] = p[i + j].y.bits64[1];
            inputKeyPinned[8 * i + j + 6 * NB_THREAD_PER_GROUP] = p[i + j].y.bits64[2];
            inputKeyPinned[8 * i + j + 7 * NB_THREAD_PER_GROUP] = p[i + j].y.bits64[3];

        }
    }

    // Fill device memory

    cudaMemcpy(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice);
    // We do not need the input pinned memory anymore
    cudaFreeHost(inputKeyPinned);
    inputKeyPinned = NULL;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
        return false;
    }

    return callKernel();
    //return true;

}


uint64_t new_2Gnx[4];
uint64_t new_2Gny[4];

bool GPUEngine::SetRandomJump(Point p) {


    new_2Gnx[0] = p.x.bits64[0];
    new_2Gnx[1] = p.x.bits64[1];
    new_2Gnx[2] = p.x.bits64[2];
    new_2Gnx[3] = p.x.bits64[3];

    new_2Gny[0] = p.y.bits64[0];
    new_2Gny[1] = p.y.bits64[1];
    new_2Gny[2] = p.y.bits64[2];
    new_2Gny[3] = p.y.bits64[3];

    cudaError_t err;

    err = cudaMemcpyToSymbol(_2Gnx, new_2Gnx, sizeof(new_2Gnx));
    if (err != cudaSuccess) {
        printf("GPUEngine: SetRandomJump _2Gnx: %s\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMemcpyToSymbol(_2Gny, new_2Gny, sizeof(new_2Gny));
    if (err != cudaSuccess) {
        printf("GPUEngine: SetRandomJump _2Gny: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
    //return callKernel();

}



bool GPUEngine::Launch(std::vector<ITEM>& addressFound, bool spinWait) {

    addressFound.clear();
    

    // Get the result


    if(spinWait) {

      // Waits for the kernel; only the hit counter is needed here, the items are copied below
      cudaMemcpy(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost);

    } else {

      // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
      cudaEvent_t evt;
      cudaEventCreate(&evt);

      //cudaMemcpy(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost, 0);

      cudaEventRecord(evt, 0);
      while (cudaEventQuery(evt) == cudaErrorNotReady) {
        // Sleep 1 ms to free the CPU
        Timer::SleepMillis(1);
      }
      cudaEventDestroy(evt);

    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
      return false;
    }

    // Look for address found
    uint32_t nbFound = outputBufferPinned[0];

    if (nbFound > maxFound) {
      // address has been lost
      if (!lostWarning) {
        printf("\nWarning, %u items lost\nHint: Search with less addresses/prefixes or increase maxFound (-m)\n", (nbFound - maxFound));
        lostWarning = true;
      }
      nbFound = maxFound;
    }

    // The kernel has ended, a standard copy can be done
    cudaMemcpy(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE32 + 1);
        ITEM it;
        it.thId = itemPtr[0];
        int16_t* ptr = (int16_t*)&(itemPtr[1]);
        it.endo = ptr[0] & 0x7FFF;
        it.mode = (ptr[0] & 0x8000) != 0;
        it.incr = ptr[1];
        it.hash = (uint8_t*)(itemPtr + 2);
        addressFound.push_back(it);
    }

    return callKernel();

}

void GPUEngine::FreeGPUEngine() {  //free gpu for Pause function

    // Ensure all operations have completed before freeing memory
    cudaDeviceSynchronize();

    ReleaseBuffers();
    initialised = false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Error freeing memory: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceReset();

}

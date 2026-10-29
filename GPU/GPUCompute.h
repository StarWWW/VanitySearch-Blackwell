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

// CUDA Kernel main function
// Compute SecpK1 keys and calculate RIPEMD160(SHA256(key)) then check address
// For the kernel, we use a 16 bits address lookup table which correspond to ~3 Base58 characters
// A second level lookup table contains 32 bits address (if used)
// (The CPU computes the full address and check the full address)
//
// We use affine coordinates for elliptic curve point (ie Z=1)



// Single full-address search: the hash160 is compared directly against constant memory
// instead of going through the 128KB prefix table (one global load per key).
__device__ __constant__ uint32_t _singleTarget = 0;
__device__ __constant__ uint32_t _targetH160[5];
// Capacity of the output buffer in items: hits past it are counted but not written.
__device__ __constant__ uint32_t _maxFound = 0;
// Multi full-address search: bloom filter over the hash160 (_bloomK probes, 0 = off).
// Probe i tests bit (h[0] + i*(h[1]|1)) & _bloomMask; the CPU removes the false positives.
__device__ __constant__ uint32_t _bloomK = 0;
__device__ __constant__ uint32_t _bloomMask;
__device__ __constant__ const uint32_t *_bloomBits;

#ifndef CHECK_INLINE
#define CHECK_INLINE __forceinline__
#endif

__device__ CHECK_INLINE void CheckPoint(uint32_t *_h, int32_t incr, address_t *address, uint32_t *lookup32, uint32_t *out) {

  uint32_t   off;
  addressl_t  l32;
  address_t   pr0;
  address_t   hit;
  uint32_t   pos;
  uint32_t   st;
  uint32_t   ed;
  uint32_t   mi;
  uint32_t   lmi;
  

    if (_singleTarget) {
        if (_h[0] != _targetH160[0]) return;
        if (_h[1] != _targetH160[1] || _h[2] != _targetH160[2] || _h[3] != _targetH160[3] || _h[4] != _targetH160[4]) return;
        goto addItem;
    }

    if (_bloomK) {
        uint32_t bit = _h[0];
        uint32_t step = _h[1] | 1;
        for (uint32_t k = 0; k < _bloomK; k++) {
            uint32_t idx = bit & _bloomMask;
            if (!(__ldg(_bloomBits + (idx >> 5)) & (1u << (idx & 31)))) return;
            bit += step;
        }
        goto addItem;
    }

    // Lookup table
    pr0 = *(address_t *)(_h);
    hit = address[pr0];

    if (hit) {

        if (lookup32) {
            off = lookup32[pr0];
            l32 = _h[0];
            st = off;
            ed = off + hit - 1;
            while (st <= ed) {
                mi = (st + ed) / 2;
                lmi = lookup32[mi];
                if (l32 < lmi) {
                    ed = mi - 1;
                }
                else if (l32 == lmi) {
                    // found
                    goto addItem;
                }
                else {
                    st = mi + 1;
                }
            }
            return;
        }

    addItem:

        pos = atomicAdd(out, 1);
        if (pos < _maxFound) {
            uint32_t   tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            out[pos * ITEM_SIZE32 + 1] = tid;
            out[pos * ITEM_SIZE32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(1 << 15);
            out[pos * ITEM_SIZE32 + 3] = _h[0];
            out[pos * ITEM_SIZE32 + 4] = _h[1];
            out[pos * ITEM_SIZE32 + 5] = _h[2];
            out[pos * ITEM_SIZE32 + 6] = _h[3];
            out[pos * ITEM_SIZE32 + 7] = _h[4];
        }

    }

}

 //-----------------------------------------------------------------------------------------

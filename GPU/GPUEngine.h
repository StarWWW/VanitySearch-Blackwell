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

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include "../SECP256k1.h"

#define SEARCH_COMPRESSED 0
#define SEARCH_UNCOMPRESSED 1
#define SEARCH_BOTH 2


#define ITEM_SIZE 28
#define ITEM_SIZE32 (ITEM_SIZE/4)
#define _64K 65536

static const char *searchModes[] = {"Compressed","Uncompressed","Compressed or Uncompressed"};

// Number of key per thread (must be a multiple of GRP_SIZE) per kernel call

	
//typedef uint16_t address_t;
typedef uint16_t address_t;
typedef uint32_t addressl_t;

typedef struct {
  uint32_t thId;
  int16_t  incr;
  int16_t  endo;
  uint8_t  *hash;
  bool mode;
} ITEM;

// Second level lookup
typedef struct {
  address_t sAddress;
  std::vector<addressl_t> lAddresses;
} LADDRESS;

class GPUEngine {

public:

  GPUEngine(int gpuId, uint32_t maxFound);
  ~GPUEngine();
  void FreeGPUEngine();
  bool SetAddress(std::vector<address_t> addresses);
  bool SetAddress(std::vector<LADDRESS> addresses,uint32_t totalAddress);
  bool SetSingleTarget(const uint8_t *hash160);
  bool SetBloom(const std::vector<const uint8_t *> &hash160s);
  bool SetKeys(Point *p);
  bool SetRandomJump(Point p);
  void SetSearchMode(int searchMode);
  void SetSearchType(int searchType);
  bool Launch(std::vector<ITEM> &addressFound,bool spinWait=false);
  int GetNbThread();
  int GetGroupSize();
  int GetStepSize();
  bool IsInitialised();

  std::string deviceName;

  static void PrintCudaInfo();
  static void GenerateCode(Secp256K1 *secp, int size);

private:

  bool callKernel();
  void ReleaseBuffers();

  int nbThread;
  address_t *inputAddress;
  address_t *inputAddressPinned;
  uint32_t *inputAddressLookUp;
  uint32_t *inputAddressLookUpPinned;
  uint32_t *bloomBits;
  uint64_t *inputKey;
  uint64_t *inputKeyPinned;
  uint32_t *outputBuffer;
  uint32_t *outputBufferPinned;
  bool initialised;
  uint32_t searchMode;
  uint32_t searchType;
  bool lostWarning;
  uint32_t maxFound;
  uint32_t outputSize;

};

#endif // GPUENGINEH

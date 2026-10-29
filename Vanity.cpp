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

#include "Vanity.h"
#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <thread>
#include <atomic>
#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <stdio.h>
#ifdef WIN64
#include <Windows.h>
#endif

using namespace std;

VanitySearch::VanitySearch(Secp256K1* secp, vector<std::string>& inputAddresses, int searchMode,
	bool stop, string outputFile, uint32_t maxFound, BITCRACK_PARAM* bc):inputAddresses(inputAddresses) 
{
	this->secp = secp;
	this->searchMode = searchMode;
	this->stopWhenFound = stop;
	this->outputFile = outputFile;
	this->numGPUs = 0;
	this->maxFound = maxFound;	
	this->searchType = -1;
	this->bc = bc;
	this->endOfSearch = false;
	this->searchStatus = SEARCH_RUNNING;
	this->nbFoundKey = 0;

	addresses.clear();

	// Create a 65536 items lookup table
	ADDRESS_TABLE_ITEM t;
	t.found = true;
	t.items = NULL;
	for (int i = 0; i < 65536; i++)
		addresses.push_back(t);
	
	// Insert addresses
	bool loadingProgress = (inputAddresses.size() > 1000);
	if (loadingProgress)
		fprintf(stdout, "[Building lookup16   0.0%%]\r");

	nbAddress = 0;
	onlyFull = true;
	vector<string> accepted;

	for (int i = 0; i < (int)inputAddresses.size(); i++)
	{
		ADDRESS_ITEM it;
		std::vector<ADDRESS_ITEM> itAddresses;

		if (initAddress(inputAddresses[i], &it)) {
			it.found = false;
			itAddresses.push_back(it);
		}

		if (itAddresses.size() > 0) 
		{
			// Add the item to all corresponding addresses in the lookup table
			for (int j = 0; j < (int)itAddresses.size(); j++) 
			{
				address_t p = itAddresses[j].sAddress;

				if (addresses[p].items == NULL) {
					addresses[p].items = new vector<ADDRESS_ITEM>();
					addresses[p].found = false;
					usedAddress.push_back(p);
				}
				(*addresses[p].items).push_back(itAddresses[j]);
			}
			onlyFull &= it.isFull;
			nbAddress++;
			if (accepted.size() < 10)
				accepted.push_back(inputAddresses[i]);
		}

		if (loadingProgress && i % 1000 == 0)
			fprintf(stdout, "[Building lookup16 %5.1f%%]\r", (((double)i) / (double)(inputAddresses.size() - 1)) * 100.0);
	}

	if (loadingProgress)
		fprintf(stdout, "\n");

	if (nbAddress == 0) 
	{
		fprintf(stderr, "[ERROR] VanitySearch: nothing to search !\n");
		exit(1);
	}

	// Second level lookup
	uint32_t unique_sAddress = 0;
	uint32_t minI = 0xFFFFFFFF;
	uint32_t maxI = 0;
	for (int i = 0; i < (int)addresses.size(); i++) 
	{
		
		if (addresses[i].items) 
		{
			
			LADDRESS lit;
			lit.sAddress = i;
			if (addresses[i].items) 
			{
				for (int j = 0; j < (int)addresses[i].items->size(); j++) 
				{
					lit.lAddresses.push_back((*addresses[i].items)[j].lAddress);
					
				}
			}

			sort(lit.lAddresses.begin(), lit.lAddresses.end());
			usedAddressL.push_back(lit);
			if ((uint32_t)lit.lAddresses.size() > maxI) maxI = (uint32_t)lit.lAddresses.size();
			if ((uint32_t)lit.lAddresses.size() < minI) minI = (uint32_t)lit.lAddresses.size();
			unique_sAddress++;
		}

		if (loadingProgress && (i % 4096) == 0)
			fprintf(stdout, "[Building lookup32 %.1f%%]\r", ((double)i * 100.0) / (double)addresses.size());
	}

	if (loadingProgress)
		fprintf(stdout, "\n");
	
	string searchInfo = string(searchModes[searchMode]);
	if (nbAddress < 10) 
	{	
		for (size_t i = 0; i < accepted.size(); i++)
		{
			fprintf(stdout, "Search: %s [%s]\n", accepted[i].c_str(), searchInfo.c_str());
		}
	}
	else 
	{		
		fprintf(stdout, "Search: %d (Lookup size %d,[%d,%d]) [%s]\n", nbAddress, unique_sAddress, minI, maxI, searchInfo.c_str());
	}

	// Constant for endomorphism
	// if a is a nth primitive root of unity, a^-1 is also a nth primitive root.
	// beta^3 = 1 mod p implies also beta^2 = beta^-1 mop (by multiplying both side by beta^-1)
	// (beta^3 = 1 mod p),  beta2 = beta^-1 = beta^2
	// (lambda^3 = 1 mod n), lamba2 = lamba^-1 = lamba^2
	beta.SetBase16("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
	lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
	beta2.SetBase16("851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");
	lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

	startKey.Set(&bc->ksNext);	

	char* ctimeBuff;
	time_t now = time(NULL);
	ctimeBuff = ctime(&now);
	fprintf(stdout, "Current task START time: %s", ctimeBuff);
	fflush(stdout);
}

VanitySearch::~VanitySearch() {

	for (int i = 0; i < (int)usedAddress.size(); i++) {
		vector<ADDRESS_ITEM>* items = addresses[usedAddress[i]].items;
		if (items) {
			delete items;
			addresses[usedAddress[i]].items = NULL;
		}
	}

}

bool VanitySearch::isSingularAddress(std::string pref) {

	// check is the given address contains only 1
	bool only1 = true;
	int i = 0;
	while (only1 && i < (int)pref.length()) {
		only1 = pref.data()[i] == '1';
		i++;
	}
	return only1;
}

// Checks the host SHA-256 on a known address once: if it is miscompiled, a valid address must
// not be rejected for its checksum
static bool cpuChecksumWorks() {

	static int works = -1;
	if (works < 0) {
		std::vector<unsigned char> r;
		uint8_t chk[4];
		works = DecodeBase58("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", r) && r.size() == 25;
		if (works) {
			sha256_checksum(r.data(), 21, chk);
			works = memcmp(chk, r.data() + 21, 4) == 0;
		}
		if (!works)
			fprintf(stdout, "[WARNING] The CPU SHA-256 of this build is wrong, address checksums are not checked\n");
	}
	return works != 0;
}

bool VanitySearch::initAddress(std::string& address, ADDRESS_ITEM* it) {

	std::vector<unsigned char> result;
	string dummy1 = address;
	int nbDigit = 0;
	bool wrong = false;

	if (address.length() < 2) {
		fprintf(stdout, "Ignoring address \"%s\" (too short)\n", address.c_str());
		return false;
	}

	int aType = -1;

	switch (address.data()[0]) {
	case '1':
		aType = P2PKH;
		break;
	case '3':
		// The GPU kernel hashes the compressed public key only (P2PKH / P2WPKH),
		// a P2SH script hash can never match
		fprintf(stdout, "Ignoring address \"%s\" (P2SH addresses are not supported, use 1... or bc1q...)\n", address.c_str());
		return false;
	case 'b':
	case 'B':
		std::transform(address.begin(), address.end(), address.begin(), ::tolower);
		if (strncmp(address.c_str(), "bc1q", 4) == 0)
			aType = BECH32;
		break;
	}

	if (aType == -1) {
		fprintf(stdout, "Ignoring address \"%s\" (must start with 1 or bc1q)\n", address.c_str());
		return false;
	}

	if (searchType == -1) searchType = aType;
	if (aType != searchType) {
		fprintf(stdout, "Ignoring address \"%s\" (P2PKH or BECH32 allowed at once, not both)\n", address.c_str());
		return false;
	}

	if (aType == BECH32) {

		// BECH32
		uint8_t witprog[40];
		size_t witprog_len;
		int witver;
		const char* hrp = "bc";

		int ret = segwit_addr_decode(&witver, witprog, &witprog_len, hrp, address.c_str());

		// Try to attack a full address ?
		if (ret && witprog_len == 20) {
						
			it->isFull = true;
			memcpy(it->hash160, witprog, 20);
			it->sAddress = *(address_t*)(it->hash160);
			it->lAddress = *(addressl_t*)(it->hash160);
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;

		}

		if (address.length() < 5) {
			fprintf(stdout, "Ignoring address \"%s\" (too short, length<5 )\n", address.c_str());
			return false;
		}

		if (address.length() >= 36) {
			fprintf(stdout, "Ignoring address \"%s\" (too long, length>36 )\n", address.c_str());
			return false;
		}

		uint8_t data[64];
		memset(data, 0, 64);
		size_t data_length;
		if (!bech32_decode_nocheck(data, &data_length, address.c_str() + 4)) {
			fprintf(stdout, "Ignoring address \"%s\" (Only \"023456789acdefghjklmnpqrstuvwxyz\" allowed)\n", address.c_str());
			return false;
		}
		
		it->sAddress = *(address_t*)data;		
		it->isFull = false;
		it->lAddress = 0;
		it->address = (char*)address.c_str();
		it->addressLength = (int)address.length();

		return true;
	}
	else {

		// P2PKH
		wrong = !DecodeBase58(address, result);

		if (wrong) {
			fprintf(stdout, "Ignoring address \"%s\" (not base58 or too long: 0, I, O and l are not allowed)\n", address.c_str());
			return false;
		}

		// Try to attack a full address ?
		if (result.size() > 21) {

			// Version 0, hash160 and a checksum, which catches a mistyped address that would
			// otherwise be searched for forever
			if (result.size() != 25 || result[0] != 0) {
				fprintf(stdout, "Ignoring address \"%s\" (too long for a prefix, not a valid P2PKH address)\n", address.c_str());
				return false;
			}
			uint8_t chk[4];
			sha256_checksum(result.data(), 21, chk);
			if (memcmp(chk, result.data() + 21, 4) != 0 && cpuChecksumWorks()) {
				fprintf(stdout, "Ignoring address \"%s\" (wrong checksum, mistyped address?)\n", address.c_str());
				return false;
			}

			it->isFull = true;
			memcpy(it->hash160, result.data() + 1, 20);
			it->sAddress = *(address_t*)(it->hash160);
			it->lAddress = *(addressl_t*)(it->hash160);
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;
		}

		// Address containing only '1'
		if (isSingularAddress(address)) {

			if (address.length() > 21) {
				fprintf(stdout, "Ignoring address \"%s\" (Too much 1)\n", address.c_str());
				return false;
			}
			
			it->isFull = false;
			it->sAddress = 0;
			it->lAddress = 0;
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;
		}

		// Search for highest hash160 16bit address (most probable)
		while (result.size() < 25) {
			DecodeBase58(dummy1, result);
			if (result.size() < 25) {
				dummy1.append("1");
				nbDigit++;
			}
		}

		if (result.size() != 25) {
			fprintf(stdout, "Ignoring address \"%s\" (Invalid size)\n", address.c_str());
			return false;
		}

		it->sAddress = *(address_t*)(result.data() + 1);

		dummy1.append("1");
		DecodeBase58(dummy1, result);

		if (result.size() == 25) {
			it->sAddress = *(address_t*)(result.data() + 1);
			nbDigit++;
		}
		
		it->isFull = false;
		it->lAddress = 0;
		it->address = (char*)address.c_str();
		it->addressLength = (int)address.length();

		return true;
	}
}

// ----------------------------------------------------------------------------

void VanitySearch::output(string addr, string pAddr, string pAddrHex, std::string pubKey) {

	std::lock_guard<std::mutex> lock(outputMutex);

	// "%064s" is undefined for strings (glibc pads with spaces), pad the key explicitly
	if (pAddrHex.length() < 64)
		pAddrHex.insert(0, 64 - pAddrHex.length(), '0');

	FILE* f = stdout;
	bool needToClose = false;

	if (outputFile.length() > 0) {
		f = fopen(outputFile.c_str(), "a");
		if (f == NULL) {
			fprintf(stderr, "Cannot open %s for writing\n", outputFile.c_str());
			f = stdout;
		}
		else {
			needToClose = true;
		}
	}




	if (f != stdout)
		fprintf(f, "\nPublic Addr: %s\n", addr.c_str());	
	fprintf(stdout, "\nPublic Addr: %s\n", addr.c_str());

	switch (searchType) {
	case P2PKH:
		if (f != stdout)
			fprintf(f, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		break;
	case BECH32:
		if (f != stdout)
			fprintf(f, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
		break;
	}

	if (f != stdout)
		fprintf(f, "Priv (HEX): 0x%s\n", pAddrHex.c_str());
	fprintf(stdout, "Priv (HEX): 0x%s\n", pAddrHex.c_str());

	if (f != stdout)
		fprintf(f, "Pub  Key  : %s\n", pubKey.c_str());
	fprintf(stdout, "Pub  Key  : %s\n", pubKey.c_str());

	if (f != stdout) {
		char* ctimeBuff;
		time_t now = time(NULL);
		ctimeBuff = ctime(&now);
		fprintf(f, "Found at  : %s", ctimeBuff);
	}
	fprintf(stdout, "\n");

	if (f != stdout)
		fflush(f);
	fflush(stdout);

	if (needToClose)
		fclose(f);
}

void VanitySearch::updateFound() {

	// Check if all addresses has been found
	// Needed only if stopWhenFound is asked
	if (stopWhenFound) 	{

		bool allFound = true;
		for (int i = 0; i < (int)usedAddress.size(); i++) {
			bool iFound = true;
			address_t p = usedAddress[i];
			if (!addresses[p].found) {
				if (addresses[p].items) {
					for (int j = 0; j < (int)addresses[p].items->size(); j++) {
						iFound &= (*addresses[p].items)[j].found;
					}
				}
				addresses[usedAddress[i]].found = iFound;
			}
			allFound &= iFound;
		}

		endOfSearch = allFound;		
	}		
}

bool VanitySearch::checkPrivKey(string addr, Int& key, int32_t incr, int endomorphism, bool mode) {

	Int k(&key);	

	if (incr < 0) {
		k.Add((uint64_t)(-incr));
		k.Neg();
		k.Add(&secp->order);		
	}
	else {
		k.Add((uint64_t)incr);
	}

	// Endomorphisms
	switch (endomorphism) {
	case 1:
		k.ModMulK1order(&lambda);		
		break;
	case 2:
		k.ModMulK1order(&lambda2);		
		break;
	}

	// Check addresses
	Point p = secp->ComputePublicKey(&k);	

	string chkAddr = secp->GetAddress(searchType, mode, p);
	if (chkAddr != addr) {

		// Key may be the opposite one (negative zero or compressed key)
		k.Neg();
		k.Add(&secp->order);
		p = secp->ComputePublicKey(&k);
		
		string chkAddr = secp->GetAddress(searchType, mode, p);
		if (chkAddr != addr) {
			fprintf(stdout, "\nWarning, wrong private key generated !\n");
			fprintf(stdout, "  Addr :%s\n", addr.c_str());
			fprintf(stdout, "  Check:%s\n", chkAddr.c_str());
			fprintf(stdout, "  Endo:%d incr:%d comp:%d\n", endomorphism, incr, mode);
			// The GPU hash matched: print the candidate so that an external check can still use it
			fprintf(stdout, "  CandKey: %s\n", k.GetBase16().c_str());
			return false;
		}

	}

	output(addr, secp->GetPrivAddress(mode, k), k.GetBase16(), secp->GetPublicKeyHex(mode, p));

	return true;
}

void VanitySearch::checkAddr(int prefIdx, uint8_t* hash160, Int& key, int32_t incr, int endomorphism, bool mode) {
	
	vector<ADDRESS_ITEM>* pi = addresses[prefIdx].items;
	if (pi == NULL)
		return;


	if (onlyFull) {

		// Full addresses
		for (int i = 0; i < (int)pi->size(); i++) {

			if (stopWhenFound && (*pi)[i].found)
				continue;

			if (ripemd160_comp_hash((*pi)[i].hash160, hash160)) {

				// Found it ! Marked only once the key is verified, a failed check keeps searching
				if (checkPrivKey(secp->GetAddress(searchType, mode, hash160), key, incr, endomorphism, mode)) {
					(*pi)[i].found = true;
					nbFoundKey++;
					updateFound();
				}

			}

		}

	}
	else {
		char a[64];

		string addr = secp->GetAddress(searchType, mode, hash160);

		for (int i = 0; i < (int)pi->size(); i++) {

			if (stopWhenFound && (*pi)[i].found)
				continue;

			strncpy(a, addr.c_str(), (*pi)[i].addressLength);
			a[(*pi)[i].addressLength] = 0;

			if (strcmp((*pi)[i].address, a) == 0) {

				// Found it !
				if (checkPrivKey(addr, key, incr, endomorphism, mode)) {
					(*pi)[i].found = true;
					nbFoundKey++;
					updateFound();
				}

			}

		}

	}


}

// Keys per GPU thread. Rounded up so that the last (taskSize mod nbThread) keys of the range
// are scanned too; with a power-of-two range and thread count it is an exact division.
void VanitySearch::getThreadStep(int nbThread, Int* stepThread) {

	Int numthread;
	Int rem;

	stepThread->Set(&bc->ksFinish);
	stepThread->Sub(&bc->ksStart);
	stepThread->AddOne();
	numthread.SetInt32(nbThread);
	stepThread->Div(&numthread, &rem);
	if (!rem.IsZero())
		stepThread->AddOne();
}

void VanitySearch::getGPUStartingKeys(Int& tRangeStart, Int& tRangeEnd, int groupSize, int nbThread, Point *p, uint64_t Progress) {

	uint32_t grp_startkeys = nbThread/256;

	//New setting key by fixedpaul using addition on secp with batch modular inverse, super fast, multithreading not needed

	Int stepThread;
	getThreadStep(nbThread, &stepThread);

	Point Pdouble;
	Int kDouble;

	kDouble.Set(&stepThread);
	kDouble.Mult(grp_startkeys);
	Pdouble = secp->ComputePublicKey(&kDouble);

	Int kStart;

	kStart.Set(&stepThread);
	kStart.Mult(grp_startkeys / 2);
	kStart.Add(groupSize / 2 + Progress);
	kStart.Add(&tRangeStart);

	// Computed directly rather than as P(start) + P(offset): an affine addition cannot handle
	// the point at infinity (start = 0) nor two equal points
	p[grp_startkeys / 2] = secp->ComputePublicKey(&kStart);


	Int key_delta;
	Point* p_delta;
	p_delta = new Point[grp_startkeys / 2];

	key_delta.Set(&stepThread);



	p_delta[0] = secp->ComputePublicKey(&key_delta);
	key_delta.Add(&stepThread);
	p_delta[1] = secp->ComputePublicKey(&key_delta);

	for (size_t i = 2; i < grp_startkeys / 2; i++) {
		p_delta[i] = secp->AddDirect(p_delta[i - 1], p_delta[0]);
	}

	Int* dx;
	Int* subp;

	subp = new Int[grp_startkeys / 2 + 1];
	dx = new Int[grp_startkeys / 2 + 1];

	uint32_t j;
	//uint32_t i;

	for (size_t i = grp_startkeys / 2; i < nbThread; i += grp_startkeys) {

		double percentage = (100.0 * (double)(i + grp_startkeys / 2)) / (double)(nbThread);
		printf("Setting starting keys... [%.2f%%] \r", percentage);
		fflush(stdout);


		for (j = 0; j < grp_startkeys / 2; j++) {
			dx[j].ModSub(&p_delta[j].x, &p[i].x);
		}
		dx[grp_startkeys / 2].ModSub(&Pdouble.x, &p[i].x);

		Int newValue;
		Int inverse;

		subp[0].Set(&dx[0]);
		for (size_t j = 1; j < grp_startkeys / 2 + 1; j++) {
			subp[j].ModMulK1(&subp[j - 1], &dx[j]);
		}

		// Do the inversion
		inverse.Set(&subp[grp_startkeys / 2]);
		inverse.ModInv();

		for (j = grp_startkeys / 2; j > 0; j--) {
			newValue.ModMulK1(&subp[j - 1], &inverse);
			inverse.ModMulK1(&dx[j]);
			dx[j].Set(&newValue);
		}

		dx[0].Set(&inverse);

		Int _s;
		Int _p;
		Int dy;
		Int syn;
		syn.Set(&p[i].y);
		syn.ModNeg();



		for (j = 0; j < grp_startkeys / 2 - 1; j++) {

			dy.ModSub(&p_delta[j].y, &p[i].y);
			_s.ModMulK1(&dy, &dx[j]);

			_p.ModSquareK1(&_s);

			p[i + j + 1].x.ModSub(&_p, &p[i].x);
			p[i + j + 1].x.ModSub(&p_delta[j].x);

			p[i + j + 1].y.ModSub(&p_delta[j].x, &p[i + j + 1].x);
			p[i + j + 1].y.ModMulK1(&_s);
			p[i + j + 1].y.ModSub(&p_delta[j].y);

			dy.ModSub(&syn, &p_delta[j].y);
			_s.ModMulK1(&dy, &dx[j]);

			_p.ModSquareK1(&_s);

			p[i - j - 1].x.ModSub(&_p, &p[i].x);
			p[i - j - 1].x.ModSub(&p_delta[j].x);

			p[i - j - 1].y.ModSub(&p[i - j - 1].x, &p_delta[j].x);
			p[i - j - 1].y.ModMulK1(&_s);
			p[i - j - 1].y.ModSub(&p_delta[j].y, &p[i - j - 1].y);
		}

		dy.ModSub(&syn, &p_delta[j].y);
		_s.ModMulK1(&dy, &dx[j]);

		_p.ModSquareK1(&_s);


		p[i - j - 1].x.ModSub(&_p, &p[i].x);
		p[i - j - 1].x.ModSub(&p_delta[j].x);

		p[i - j - 1].y.ModSub(&p[i - j - 1].x, &p_delta[j].x);
		p[i - j - 1].y.ModMulK1(&_s);
		p[i - j - 1].y.ModSub(&p_delta[j].y, &p[i - j - 1].y);

		if (i + grp_startkeys < nbThread) {

			dy.ModSub(&Pdouble.y, &p[i].y);
			_s.ModMulK1(&dy, &dx[grp_startkeys / 2]);

			_p.ModSquareK1(&_s);

			p[i + grp_startkeys].x.ModSub(&_p, &p[i].x);
			p[i + grp_startkeys].x.ModSub(&Pdouble.x);

			p[i + grp_startkeys].y.ModSub(&Pdouble.x, &p[i + grp_startkeys].x);
			p[i + grp_startkeys].y.ModMulK1(&_s);
			p[i + grp_startkeys].y.ModSub(&Pdouble.y);
		}
	}

	delete[] subp;
	delete[] dx;
	delete[] p_delta;
}

void VanitySearch::FindKeyGPU(TH_PARAM* ph) {

	bool ok = true;

	double t0;
	double ttot = t_Paused;

	// Global init
	GPUEngine g(ph->gpuId, maxFound);
	if (!g.IsInitialised()) {
		fprintf(stderr, "[ERROR] GPU #%d initialisation failed\n", ph->gpuId);
		searchStatus = SEARCH_ERROR;
		endOfSearch = true;
		ph->isRunning = false;
		return;
	}

	int numThreadsGPU = g.GetNbThread();
	int STEP_SIZE = g.GetStepSize();

	// A backup only maps to the same key layout: same range (checked when loaded), thread count and
	// keys per launch (VS_GRP)
	if (!randomMode && backupStepSize != 0 && backupStepSize != (uint32_t)STEP_SIZE) {
		fprintf(stderr, "[ERROR] The backup was made with %u keys per thread and launch, this build uses %d (VS_GRP).\n"
			"        Resume with a build of the same group size or start again without -backup.\n",
			backupStepSize, STEP_SIZE);
		searchStatus = SEARCH_ERROR;
		endOfSearch = true;
		ph->isRunning = false;
		return;
	}
	if (!randomMode && backupThreads != 0 && backupThreads != (uint32_t)numThreadsGPU) {
		fprintf(stderr, "[ERROR] The backup was made with %u GPU threads, this GPU runs %d. The progress cannot be\n"
			"        mapped to this GPU: resume on the original GPU model or start again without -backup.\n",
			backupThreads, numThreadsGPU);
		searchStatus = SEARCH_ERROR;
		endOfSearch = true;
		ph->isRunning = false;
		return;
	}

	Point* publicKeys = new Point[numThreadsGPU];
	vector<ITEM> found;

	Point RandomJump_P;
	Int RandomJump_K;
	Int RandomJump_K_last;
	Int RandomJump_K_tot;
	RandomJump_K.SetInt32(STEP_SIZE);
	RandomJump_K_last.SetInt32(0);
	RandomJump_K_tot.SetInt32(0);

	// -random jumps: a 64-bit generator seeded with 256 bits, so that processes (one per GPU) and
	// restarts never share a sequence
	std::random_device rd;
	std::seed_seq seed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), (unsigned)time(NULL) };
	std::mt19937_64 rng(seed);

	fprintf(stdout, "GPU: %s\n", g.deviceName.c_str());
	fflush(stdout);

	g.SetSearchMode(searchMode);
	g.SetSearchType(searchType);
	if (onlyFull && nbAddress == 1) {
		for (int i = 0; i < (int)addresses.size(); i++) {
			if (addresses[i].items) {
				ok = g.SetSingleTarget((*addresses[i].items)[0].hash160);
				break;
			}
		}
	}
	else if (onlyFull) {
		ok = g.SetAddress(usedAddressL, nbAddress);
	}
	else {
		ok = g.SetAddress(usedAddress);
	}

	Int stepThread;
	Int taskSize;

	taskSize.Set(&bc->ksFinish);
	taskSize.Sub(&bc->ksStart);
	taskSize.AddOne();
	getThreadStep(numThreadsGPU, &stepThread);

	Int privkey;
	Int part_key;
	Int keycount;
	Int threadDone;

	t0 = Timer::get_tick();

	// In -random mode the reported keys do not include a sequential offset, so none may be applied
	uint64_t progress = randomMode ? 0 : idxcount * (uint64_t)STEP_SIZE;
	if (ok) {
		getGPUStartingKeys(bc->ksStart, bc->ksFinish, g.GetGroupSize(), numThreadsGPU, publicKeys, progress);
		ok = g.SetKeys(publicKeys);
	}
	delete[] publicKeys;

	printf("Starting keys set in %.2f seconds \n", Timer::get_tick() - t0);
	fflush(stdout);

	ph->hasStarted = true;

	if (ok) {
		printf("GPU Started ! \r");
		fflush(stdout);
	}

	t0 = Timer::get_tick();

	double keysPerLaunch = (double)STEP_SIZE * (double)numThreadsGPU;
	double keysDone = keysPerLaunch * (double)idxcount;
	double keysPrev = keysDone;
	double tprev = ttot;
	double speedAvg = 0.0;
	uint64_t launches = 0;

	while (ok && !endOfSearch) {

		if (stopRequested) {
			searchStatus = SEARCH_STOPPED;
			break;
		}

		if (Pause) {
			printf("\nPausing...\n");
			fflush(stdout);
			// The kernel in flight is not counted, it is scanned again on resume
			t_Paused = ttot;
			if (!randomMode)
				saveBackup(numThreadsGPU, STEP_SIZE, ttot, ph->gpuId);
			g.FreeGPUEngine();
			printf("Paused (GPU released) - press p to resume, Ctrl+C to quit\n");
			fflush(stdout);
			searchStatus = SEARCH_PAUSED;
			Paused = true;
			break;
		}

		if (randomMode) {
			RandomJump_K_last.Set(&RandomJump_K);
			RandomJump_K_tot.Add(&RandomJump_K);

			RandomJump_K.SetInt32(0);
			for (int i = 0; i < 4; i++)
				RandomJump_K.bits64[i] = rng();
			RandomJump_K.Mod(&stepThread);
			RandomJump_K.Sub(&RandomJump_K_tot);

			if (RandomJump_K.IsNegative()) {
				RandomJump_K.Neg();
				RandomJump_P = secp->ComputePublicKey(&RandomJump_K);
				RandomJump_P.y.ModNeg();
				RandomJump_K.Neg();
			}
			else {
				RandomJump_P = secp->ComputePublicKey(&RandomJump_K);
			}

			ok = g.SetRandomJump(RandomJump_P);
			if (!ok) break;
		}

		ok = g.Launch(found, true);
		if (!ok) break;
		idxcount += 1;
		launches++;

		ttot = Timer::get_tick() - t0 + t_Paused;

		// Offset of the kernel that just completed, within each thread's slice
		keycount.SetInt32(0);
		keycount.Add((uint64_t)(idxcount - 1));
		keycount.Mult(STEP_SIZE);

		for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {

			ITEM it = found[i];
			part_key.Set(&stepThread);
			part_key.Mult(it.thId);

			privkey.Set(&bc->ksStart);
			privkey.Add(&part_key);

			if (randomMode) {
				privkey.Add(&RandomJump_K_tot);
				privkey.Sub(&RandomJump_K_last);
			}
			else {
				privkey.Add(&keycount);
			}

			checkAddr(*(address_t*)(it.hash), it.hash, privkey, it.incr, it.endo, it.mode);
		}

		if (!randomMode && idxcount % 60 == 0)
			saveBackup(numThreadsGPU, STEP_SIZE, ttot, ph->gpuId);

		keysDone = keysPerLaunch * (double)idxcount;
		double speed = 0.0;
		if (ttot > tprev)
			speed = (keysDone - keysPrev) / (ttot - tprev) / 1000000.0; // speed in Mkey/s
		// The first launch also includes the start-up, leave it out of the average
		if (launches > 1)
			speedAvg = (speedAvg == 0.0) ? speed : 0.9 * speedAvg + 0.1 * speed;
		PrintStats(keysDone, (speedAvg > 0.0) ? speedAvg : speed, ttot, taskSize.ToDouble());
		keysPrev = keysDone;
		tprev = ttot;

		// Each thread scans its own slice of stepThread keys
		threadDone.SetInt32(0);
		threadDone.Add((uint64_t)idxcount);
		threadDone.Mult(STEP_SIZE);
		if (!randomMode && threadDone.IsGreaterOrEqual(&stepThread))
		{
			double avg_speed = (ttot > 0.0) ? keysDone / (ttot * 1000000.0) : 0.0; // Avg speed in MK/s
			printf("\n");
			printf("Range Finished! - Average Speed: %.1f [MK/s] - Found: %d   \n", avg_speed, nbFoundKey);
			fflush(stdout);

			char* ctimeBuff;
			time_t now = time(NULL);
			ctimeBuff = ctime(&now);
			printf("Current task END time: %s", ctimeBuff);

			saveBackup(numThreadsGPU, STEP_SIZE, ttot, ph->gpuId);
			searchStatus = SEARCH_FINISHED;
			endOfSearch = true;
		}
	}

	if (!ok) {
		fprintf(stderr, "\n[ERROR] GPU #%d failed, search aborted\n", ph->gpuId);
		searchStatus = SEARCH_ERROR;
	}
	else if (searchStatus == SEARCH_STOPPED) {
		printf("\nStopped by user.\n");
		if (!randomMode) {
			saveBackup(numThreadsGPU, STEP_SIZE, ttot, ph->gpuId);
			printf("Progress saved to %s, resume with -backup\n", backupFileName(ph->gpuId).c_str());
		}
	}
	else if (searchStatus == SEARCH_RUNNING && endOfSearch) {
		// -stop and every address has been found
		if (!randomMode)
			saveBackup(numThreadsGPU, STEP_SIZE, ttot, ph->gpuId);
		searchStatus = SEARCH_ALL_FOUND;
	}
	t_Paused = ttot;
	fflush(stdout);

	endOfSearch = true;
	ph->isRunning = false;
}


static void formatDuration(char* buf, size_t len, double t) {

	if (!(t >= 0.0) || t > 3.15e13) {             // negative, NaN or beyond a million years
		snprintf(buf, len, "--:--:--");
		return;
	}
	double years = t / (365.25 * 86400.0);
	if (years >= 100.0) {
		snprintf(buf, len, "%.3g years", years);
		return;
	}
	uint64_t s = (uint64_t)t;
	uint64_t d = s / 86400;
	int h = (int)((s % 86400) / 3600);
	int m = (int)((s % 3600) / 60);
	int sec = (int)(s % 60);
	if (d > 0)
		snprintf(buf, len, "%llud %02d:%02d:%02d", (unsigned long long)d, h, m, sec);
	else
		snprintf(buf, len, "%02d:%02d:%02d", h, m, sec);
}

void VanitySearch::PrintStats(double keysDone, double speed, double ttot, double taskSize) {

	double perc = (taskSize > 0.0) ? 100.0 * keysDone / taskSize : 0.0;
	// The last launch can run past the end of a small range
	if (!randomMode && perc > 100.0)
		perc = 100.0;
	double log_keys = (keysDone >= 1.0) ? log2(keysDone) : 0.0;
	double bkeys = keysDone / 1000000000.0;

	char run[64];
	formatDuration(run, sizeof(run), ttot);

	if (randomMode) {
		printf("%.1f MK/s - %.0f BKeys - 2^%.2f [%.2f%%] - RUN: %s - Found: %d     ",
			speed, bkeys, log_keys, perc, run, nbFoundKey);
	}
	else {
		char end[64];
		double left = taskSize - keysDone;
		if (left < 0.0) left = 0.0;
		formatDuration(end, sizeof(end), (speed > 0.0) ? left / (speed * 1000000.0) : -1.0);
		printf("%.1f MK/s - %.0f BKeys - 2^%.2f [%.2f%%] - RUN: %s|END: %s - Found: %d     ",
			speed, bkeys, log_keys, perc, run, end, nbFoundKey);
	}

	printf("\r");
	fflush(stdout);
}


// ----------------------------------------------------------------------------
// Backup file

static const char BACKUP_MAGIC[8] = { 'V','S','B','A','C','K','2','\n' };

std::string backupFileName(int gpuId) {
	return "VSbackup_gpu" + std::to_string(gpuId) + ".dat";
}

static void writeKey(std::ofstream& f, const Int& k) {
	for (int i = 0; i < 4; i++)
		f.write(reinterpret_cast<const char*>(&k.bits64[i]), sizeof(uint64_t));
}

static void readKey(std::ifstream& f, Int& k) {
	k.SetInt32(0);
	for (int i = 0; i < 4; i++)
		f.read(reinterpret_cast<char*>(&k.bits64[i]), sizeof(uint64_t));
}

bool readBackup(const std::string& fileName, BACKUP_STATE* st) {

	std::ifstream f(fileName, std::ios::binary);
	if (!f)
		return false;

	f.seekg(0, std::ios::end);
	std::streamoff size = f.tellg();
	f.seekg(0, std::ios::beg);

	st->legacy = false;
	st->nbThread = 0;
	st->stepSize = 0;
	st->start.SetInt32(0);
	st->finish.SetInt32(0);

	if (size == (std::streamoff)(sizeof(int32_t) + sizeof(double))) {
		// Original VanitySearch-Bitcrack format: int idxcount, double time
		int32_t idx = 0;
		f.read(reinterpret_cast<char*>(&idx), sizeof(idx));
		f.read(reinterpret_cast<char*>(&st->elapsed), sizeof(st->elapsed));
		if (!f || idx < 0)
			return false;
		st->idxcount = (uint64_t)idx;
		st->legacy = true;
		return true;
	}

	char magic[8];
	f.read(magic, sizeof(magic));
	if (!f || memcmp(magic, BACKUP_MAGIC, sizeof(magic)) != 0)
		return false;
	f.read(reinterpret_cast<char*>(&st->idxcount), sizeof(st->idxcount));
	f.read(reinterpret_cast<char*>(&st->elapsed), sizeof(st->elapsed));
	readKey(f, st->start);
	readKey(f, st->finish);
	f.read(reinterpret_cast<char*>(&st->nbThread), sizeof(st->nbThread));
	f.read(reinterpret_cast<char*>(&st->stepSize), sizeof(st->stepSize));
	return (bool)f;
}

bool writeBackup(const std::string& fileName, const BACKUP_STATE* st) {

	// Written next to the target and renamed over it, so an interrupted write never leaves a
	// truncated backup behind
	std::string tmp = fileName + ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
		if (!f)
			return false;
		f.write(BACKUP_MAGIC, sizeof(BACKUP_MAGIC));
		f.write(reinterpret_cast<const char*>(&st->idxcount), sizeof(st->idxcount));
		f.write(reinterpret_cast<const char*>(&st->elapsed), sizeof(st->elapsed));
		writeKey(f, st->start);
		writeKey(f, st->finish);
		f.write(reinterpret_cast<const char*>(&st->nbThread), sizeof(st->nbThread));
		f.write(reinterpret_cast<const char*>(&st->stepSize), sizeof(st->stepSize));
		f.flush();
		if (!f)
			return false;
	}

#ifdef WIN64
	return MoveFileExA(tmp.c_str(), fileName.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
	return rename(tmp.c_str(), fileName.c_str()) == 0;
#endif
}

void VanitySearch::saveBackup(int nbThread, int stepSize, double elapsed, int gpuid) {

	BACKUP_STATE st;
	st.idxcount = idxcount;
	st.elapsed = elapsed;
	st.start.Set(&bc->ksStart);
	st.finish.Set(&bc->ksFinish);
	st.nbThread = (uint32_t)nbThread;
	st.stepSize = (uint32_t)stepSize;
	st.legacy = false;

	std::string filename = backupFileName(gpuid);
	if (!writeBackup(filename, &st))
		std::cerr << "\nError writing backup file: " << filename << "\n";
}

static void _FindKeyGPUThread(TH_PARAM* p) {
	p->obj->FindKeyGPU(p);
}

int VanitySearch::Search(int gpuId) {

	endOfSearch = false;
	searchStatus = SEARCH_RUNNING;
	// One GPU per process: run one instance per GPU with its own -gpuId and range
	numGPUs = 1;

	TH_PARAM* params = new TH_PARAM[numGPUs]();
	std::thread* threads = new std::thread[numGPUs];

	// Launch GPU threads
	for (int i = 0; i < numGPUs; i++) {
		params[i].obj = this;
		params[i].threadId = i;
		params[i].isRunning = true;
		params[i].hasStarted = false;
		params[i].gpuId = gpuId;

		threads[i] = std::thread(_FindKeyGPUThread, params + i);
	}

	// The GPU thread owns the engine and releases it before returning, joining it guarantees
	// that the GPU is free again (pause) and that nothing still touches params
	for (int i = 0; i < numGPUs; i++)
		threads[i].join();

	delete[] threads;
	delete[] params;

	return searchStatus;
}

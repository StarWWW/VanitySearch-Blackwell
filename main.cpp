/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Modified in 2026 for NVIDIA Blackwell (sm_120) throughput and robustness, see the git history.
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


#include <sstream>
#include "Timer.h"
#include "Vanity.h"
#include "SECP256k1.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <iostream>
#include <csignal>
#include <cstdlib>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cmath>

#if defined(_WIN32) || defined(_WIN64)
#include <conio.h> // For _kbhit and _getch on Windows
#include <io.h>
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

std::atomic<bool> Pause(false);
std::atomic<bool> Paused(false);
std::atomic<bool> stopRequested(false);
std::atomic<bool> stopMonitorKey(false);
uint64_t idxcount = 0;
double t_Paused = 0;
uint32_t backupThreads = 0;
uint32_t backupStepSize = 0;
bool randomMode = false;

#if defined(_WIN32) || defined(_WIN64)

static bool stdinIsTerminal() {
	return _isatty(_fileno(stdin)) != 0;
}

static void restoreTerminal() {
}

void monitorKeypress() {
	while (!stopMonitorKey) {
		Timer::SleepMillis(10);
		if (_kbhit()) { // Check if a key is pressed
			int ch = _getch(); // Get the pressed key
			if (ch == 'p' || ch == 'P') {
				Pause = !(Pause);
			}
		}
	}
}

#else

static struct termios savedTermios;
static int savedFlags = -1;
static std::atomic<bool> terminalModified(false);

// A background job (cmd &) that touches the terminal would be stopped by SIGTTOU/SIGTTIN
static bool stdinIsTerminal() {
	return isatty(STDIN_FILENO) != 0 && tcgetpgrp(STDIN_FILENO) == getpgrp();
}

// Async-signal-safe (tcsetattr and fcntl only), also called from the signal handler.
// The flags live on the open file description shared with the shell, so leaving stdin raw or
// non-blocking would break the shell after exit.
static void restoreTerminal() {
	if (terminalModified.exchange(false)) {
		tcsetattr(STDIN_FILENO, TCSANOW, &savedTermios);
		if (savedFlags != -1)
			fcntl(STDIN_FILENO, F_SETFL, savedFlags);
	}
}

static void setupTerminal() {
	if (tcgetattr(STDIN_FILENO, &savedTermios) != 0)
		return;
	savedFlags = fcntl(STDIN_FILENO, F_GETFL, 0);
	struct termios raw = savedTermios;
	raw.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo, keep ISIG for Ctrl+C
	terminalModified = true;
	tcsetattr(STDIN_FILENO, TCSANOW, &raw);
	if (savedFlags != -1)
		fcntl(STDIN_FILENO, F_SETFL, savedFlags | O_NONBLOCK);
}

void monitorKeypress() {

	setupTerminal();

	while (!stopMonitorKey) {
		Timer::SleepMillis(10);
		char ch;
		if (read(STDIN_FILENO, &ch, 1) > 0) {
			if (ch == 'p' || ch == 'P') {
				Pause = !Pause;
			}
		}
	}

	restoreTerminal();
}

#endif

static void onSignal(int sig) {
	if (stopRequested) {
		// Second Ctrl+C: leave immediately
		restoreTerminal();
		std::_Exit(130);
	}
	stopRequested = true;
	signal(sig, onSignal);
}


#define RELEASE "1.2 (VanitySearch-Bitcrack 2.2 by FixedPaul, tuned for Blackwell)"

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {

	printf("VanitySearch [-v] [-h] [-l] [-gpuId N] [-i inputfile] [-o outputfile] [-start HEX]\n");
	printf("             [-range BITS | -end HEX] [-m N] [-stop] [-random] [-backup] [address ...]\n\n");
	printf(" -v            Print version\n");
	printf(" -h            Print this help\n");
	printf(" -l            List the CUDA devices and exit\n");
	printf(" -gpuId N      GPU to use, default 0. One GPU per process: run one instance per GPU\n");
	printf(" -i file       Read the addresses to search from a file (one per line, # for comments)\n");
	printf(" -o file       Also append the found keys to this file\n");
	printf(" -start HEX    First private key of the range, default 1\n");
	printf(" -range BITS   Range size, start -> start + 2^BITS - 1 (1..256), default 30\n");
	printf(" -end HEX      Last private key of the range (inclusive), instead of -range\n");
	printf(" -m N          Max hits stored per kernel launch, default 262144\n");
	printf(" -stop         Stop when all addresses are found\n");
	printf(" -random       Random mode: at each step every GPU thread jumps to the same random offset\n");
	printf("               within its own slice of the range\n");
	printf(" -backup       Resume a sequential search from VSbackup_gpu<N>.dat (written every 60 launches,\n");
	printf("               on pause and on Ctrl+C). The file must belong to the same range and GPU model.\n\n");
	printf("Addresses: compressed P2PKH (1...) or P2WPKH (bc1q...), full addresses or prefixes.\n");
	printf("Keys while running: p = pause/resume (releases the GPU), Ctrl+C = save progress and quit.\n");
}

[[noreturn]] static void usageError(const char* fmt, const char* arg) {
	fprintf(stderr, "[ERROR] ");
	fprintf(stderr, fmt, arg);
	fprintf(stderr, "\nUse -h for help\n");
	exit(1);
}

static const char* needValue(int argc, char* argv[], int a) {
	if (a + 1 >= argc || argv[a + 1][0] == 0)
		usageError("%s needs a value", argv[a]);
	return argv[a + 1];
}

static long long getInt(const char* name, const char* v, long long minV, long long maxV) {

	long long r = 0;
	size_t used = 0;
	try {
		r = std::stoll(string(v), &used);
	}
	catch (std::exception&) {
		usageError("Invalid %s argument, number expected", name);
	}
	if (used != strlen(v))
		usageError("Invalid %s argument, number expected", name);
	if (r < minV || r > maxV) {
		fprintf(stderr, "[ERROR] %s must be between %lld and %lld\n", name, minV, maxV);
		exit(1);
	}
	return r;
}

// Parses a private key given in hex (optional 0x prefix, at most 64 digits)
static void getHexKey(const char* name, const char* v, Int& key) {

	string s(v);
	if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
		s = s.substr(2);
	if (s.empty() || s.size() > 64)
		usageError("Invalid %s argument, 1 to 64 hex digits expected", name);
	for (char c : s) {
		if (!isxdigit((unsigned char)c))
			usageError("Invalid %s argument, hex digits expected", name);
	}
	key.SetBase16((char*)s.c_str());
}

void parseFile(string fileName, vector<string>& lines) {

	ifstream inFile(fileName, ios::binary);
	if (!inFile) {
		fprintf(stderr, "[ERROR] ParseFile: cannot open %s %s\n", fileName.c_str(), strerror(errno));
		exit(1);
	}
	inFile.seekg(0, ios::end);
	double sz = (double)inFile.tellg();
	inFile.seekg(0, ios::beg);
	bool loadingProgress = sz > 100000;
	lines.reserve(lines.size() + (size_t)(sz / 34.0)); // a P2PKH address and its line ending

	// Parse file
	int nbLine = 0;
	double bytesRead = 0;
	string line;
	while (getline(inFile, line)) {

		bytesRead += (double)line.size() + 1;

		// Remove ending \r\n and surrounding blanks
		while (!line.empty() && isspace((unsigned char)line.back()))
			line.pop_back();
		size_t first = 0;
		while (first < line.size() && isspace((unsigned char)line[first]))
			first++;
		line.erase(0, first);

		// Skip blank lines and comments
		if (line.length() > 0 && line[0] != '#') {
			lines.push_back(line);
			nbLine++;
			if (loadingProgress && (nbLine % 50000) == 0) {
				fprintf(stdout, "[Loading input file %5.1f%%]\r", 100.0 * bytesRead / sz);
				fflush(stdout);
			}
		}
	}

	if (loadingProgress)
		fprintf(stdout, "[Loading input file 100.0%%]\n");
}

int main(int argc, char* argv[]) {

	// Global Init
	Timer::Init();

	// Browse arguments
	if (argc < 2) {
		printUsage();
		exit(1);
	}

	int a = 1;
	bool stop = false;
	bool backupMode = false;
	int searchMode = SEARCH_COMPRESSED;
	int gpuId = 0;
	vector<string> address;
	string outputFile = "";
	uint32_t maxFound = 65536 * 4;
	int range = 30;
	bool rangeSet = false;
	bool endSet = false;
	Int startKey;
	Int endKey;
	startKey.SetInt32(1);

	// bitcrack mod
	BITCRACK_PARAM bitcrack, *bc;
	bc = &bitcrack;
	Int maxKey;

	// Curve order - 1, the largest valid private key
	maxKey.SetBase16((char*)"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");

	while (a < argc) {

		if (strcmp(argv[a], "-gpuId") == 0) {
			string v = needValue(argc, argv, a);
			size_t comma = v.find(',');
			if (comma != string::npos) {
				v = v.substr(0, comma);
				fprintf(stdout, "[WARNING] One GPU per process, using GPU %s only. Run one instance per GPU.\n", v.c_str());
			}
			gpuId = (int)getInt("-gpuId", v.c_str(), 0, 255);
			a += 2;
		}
		else if (strcmp(argv[a], "-v") == 0 || strcmp(argv[a], "--version") == 0) {
			printf("%s\n", RELEASE);
			exit(0);
		}
		else if (strcmp(argv[a], "-h") == 0 || strcmp(argv[a], "--help") == 0) {
			printUsage();
			exit(0);
		}
		else if (strcmp(argv[a], "-l") == 0) {
			GPUEngine::PrintCudaInfo();
			exit(0);
		}
		else if (strcmp(argv[a], "-o") == 0) {
			outputFile = string(needValue(argc, argv, a));
			a += 2;
		}
		else if (strcmp(argv[a], "-start") == 0) {
			getHexKey("-start", needValue(argc, argv, a), startKey);
			a += 2;
		}
		else if (strcmp(argv[a], "-end") == 0) {
			getHexKey("-end", needValue(argc, argv, a), endKey);
			endSet = true;
			a += 2;
		}
		else if (strcmp(argv[a], "-i") == 0) {
			parseFile(string(needValue(argc, argv, a)), address);
			a += 2;
		}
		else if (strcmp(argv[a], "-stop") == 0) {
			stop = true;
			a++;
		}
		else if (strcmp(argv[a], "-random") == 0) {
			randomMode = true;
			a++;
		}
		else if (strcmp(argv[a], "-backup") == 0) {
			backupMode = true;
			a++;
		}
		else if (strcmp(argv[a], "-range") == 0) {
			range = (int)getInt("-range", needValue(argc, argv, a), 1, 256);
			rangeSet = true;
			a += 2;
		}
		else if (strcmp(argv[a], "-m") == 0) {
			maxFound = (uint32_t)getInt("-m", needValue(argc, argv, a), 1, 1 << 24);
			a += 2;
		}
		else if (argv[a][0] == '-') {
			usageError("Unexpected %s argument", argv[a]);
		}
		else {
			address.push_back(string(argv[a]));
			a++;
		}

	}

	fprintf(stdout, "VanitySearch-Blackwell v" RELEASE "\n");

	if (address.empty())
		usageError("%s", "No address to search, give one on the command line or use -i");
	if (rangeSet && endSet)
		usageError("%s", "Use either -range or -end, not both");
	if (backupMode && randomMode)
		usageError("%s", "-backup resumes a sequential search, it cannot be used with -random");

	// Key range
	if (startKey.IsZero()) {
		fprintf(stdout, "[keyspace]  0 is not a valid private key, starting at 1\n");
		startKey.SetInt32(1);
	}
	if (startKey.IsGreater(&maxKey)) {
		fprintf(stderr, "[ERROR] -start is beyond the curve order (max %s)\n", maxKey.GetBase16().c_str());
		exit(1);
	}
	bc->ksStart.Set(&startKey);

	if (endSet) {
		if (endKey.IsGreater(&maxKey)) {
			fprintf(stderr, "[ERROR] -end is beyond the curve order (max %s)\n", maxKey.GetBase16().c_str());
			exit(1);
		}
		if (endKey.IsLower(&startKey)) {
			fprintf(stderr, "[ERROR] -end is lower than -start\n");
			exit(1);
		}
		bc->ksFinish.Set(&endKey);
	}
	else if (range >= 256) {
		fprintf(stdout, "[keyspace]  range clipped to the curve order\n");
		bc->ksFinish.Set(&maxKey);
	}
	else {
		Int Range;
		Range.SetInt32(1);
		Range.ShiftL(range);
		Range.SubOne();
		bc->ksFinish.Set(&bc->ksStart);
		bc->ksFinish.Add(&Range);
		if (bc->ksFinish.IsGreater(&maxKey)) {
			fprintf(stdout, "[keyspace]  range clipped to the curve order\n");
			bc->ksFinish.Set(&maxKey);
		}
	}
	bc->ksNext.Set(&bc->ksStart);

	Int taskSize;
	taskSize.Set(&bc->ksFinish);
	taskSize.Sub(&bc->ksStart);
	taskSize.AddOne();

	fprintf(stdout, "[keyspace]  range=2^%.2f\n", log2(taskSize.ToDouble()));
	fprintf(stdout, "[keyspace]  start=%s\n", bc->ksStart.GetBase16().c_str());
	fprintf(stdout, "[keyspace]    end=%s\n", bc->ksFinish.GetBase16().c_str());
	if (randomMode) {
		fprintf(stdout, "Random Mode Enabled !\n");
	}
	fflush(stdout);

	// Progress
	idxcount = 0;
	t_Paused = 0;
	backupThreads = 0;
	string backupFile = backupFileName(gpuId);

	if (backupMode) {
		BACKUP_STATE st;
		if (!readBackup(backupFile, &st)) {
			fprintf(stdout, "[backup]    %s not found or unreadable, starting from the beginning\n", backupFile.c_str());
		}
		else if (st.legacy) {
			fprintf(stdout, "[backup]    %s is in the old format (no range recorded): make sure -start/-range\n"
				"            and the GPU are the same as in the original run\n", backupFile.c_str());
			idxcount = st.idxcount;
			t_Paused = st.elapsed;
		}
		else if (!st.start.IsEqual(&bc->ksStart) || !st.finish.IsEqual(&bc->ksFinish)) {
			fprintf(stderr, "[ERROR] %s belongs to another range:\n", backupFile.c_str());
			fprintf(stderr, "        start=%s\n", st.start.GetBase16().c_str());
			fprintf(stderr, "          end=%s\n", st.finish.GetBase16().c_str());
			fprintf(stderr, "        Use the same -start/-range, or run without -backup.\n");
			exit(1);
		}
		else {
			idxcount = st.idxcount;
			t_Paused = st.elapsed;
			backupThreads = st.nbThread;
			backupStepSize = st.stepSize;
		}
		if (idxcount > 0) {
			fprintf(stdout, "[backup]    resuming after %llu launches (%.1f s of previous run time)\n",
				(unsigned long long)idxcount, t_Paused);
		}
		fprintf(stdout, "Backup enabled ! \n");
	}
	else if (!randomMode) {
		// The new run writes its own progress to the same file: keep the previous one aside
		BACKUP_STATE st;
		if (readBackup(backupFile, &st)) {
			string prev = backupFile + ".prev";
			remove(prev.c_str());
			if (rename(backupFile.c_str(), prev.c_str()) == 0)
				fprintf(stdout, "[backup]    previous %s moved to %s (use -backup to resume a search)\n",
					backupFile.c_str(), prev.c_str());
		}
	}

	// Init SecpK1
	Secp256K1* secp = new Secp256K1();
	secp->Init();

	signal(SIGINT, onSignal);
	signal(SIGTERM, onSignal);
#ifdef SIGBREAK
	signal(SIGBREAK, onSignal);   // Ctrl+Break / console close on Windows
#endif
	atexit(restoreTerminal);

	// Only an interactive console can pause; a pipe or /dev/null as stdin must stay untouched
	std::thread inputThread;
	if (stdinIsTerminal())
		inputThread = std::thread(monitorKeypress);

	int status = SEARCH_RUNNING;
	Pause = false;

	// One search object for the whole run: a resume after a pause keeps the addresses already
	// found, so -stop still stops once every address has been found
	VanitySearch* v = new VanitySearch(secp, address, searchMode, stop, outputFile, maxFound, bc);

	while (true) {

		Paused = false;
		status = v->Search(gpuId);

		if (status != SEARCH_PAUSED)
			break;

		while (Pause && !stopRequested) {
			Timer::SleepMillis(100);
		}
		if (stopRequested) {
			fprintf(stdout, "\nStopped by user while paused.\n");
			status = SEARCH_STOPPED;
			break;
		}
		fprintf(stdout, "\nResuming...\n");
	}

	stopMonitorKey = true;
	if (inputThread.joinable()) {
		inputThread.join();
	}
	restoreTerminal();

	int nbFound = v->GetNbFound();
	delete v;
	if (nbFound > 0)
		fprintf(stdout, "\nFound %d key(s)%s%s\n", nbFound, outputFile.empty() ? "" : ", saved to ", outputFile.c_str());
	fflush(stdout);

	delete secp;

	return (status == SEARCH_ERROR) ? 1 : 0;
}

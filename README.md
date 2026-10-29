# VanitySearch-Blackwell

Based on [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul/VanitySearch-Bitcrack),
which is in turn based on [Jean-Luc Pons' VanitySearch](https://github.com/JeanLucPons/VanitySearch). It is
tuned for NVIDIA Blackwell GPUs (sm_120) through profiling.

The tool searches a private-key range for a compressed P2PKH (`1...`) or P2WPKH (`bc1q...`) address
or prefix. The command line is upstream's plus a few additions (see [Usage](#usage)). The GPU kernel
needs **10.8% fewer SM cycles per key**, and every change was A/B measured and verified.

| GPU | Original | VanitySearch-Blackwell |
|---|---|---|
| SM cycles per key (sm_120, locked clocks) | 1.000 | **0.892** |
| RTX 5050 Laptop (80 W power cap) | 1070 MK/s | **1166 MK/s (+9.0%)** |
| RTX PRO 6000 Blackwell (Google Colab) | 8625 MK/s | **9479 MK/s (+9.9%)** |

How it was done, including the experiments that failed, is in **[OPTIMIZATION.md](OPTIMIZATION.md)**.
In short, the kernel is bound by the ALU pipe, which is filled by the rotations and boolean functions
of SHA-256 and RIPEMD-160. The FMA pipe is two-thirds idle. The wins came from moving the work that
*can* move (reduction multiplies, 3-input additions) to the FMA pipe, plus a single-target fast path
and a better launch shape.

## What changed

* A single full-address search compares hash160 against constant memory instead of a 128 KB lookup table.
* The hash160 and check functions are inlined.
* The 512→320-bit field reduction uses multiplies instead of a shift/add expansion.
* SHA-256 and RIPEMD-160 additions are expressed as `IMAD x·1 + y` (FMA pipe).
* Launches use 128 threads per block with twice the blocks (same threads per launch).
* Build-time tuning options are added (`VS_*`, see OPTIMIZATION.md).
* Tooling is added: a Windows build script, a planted-key self-test, A/B and Nsight Compute
  cycle benchmarks, and pipe microbenchmarks (`bench/`).
* The Makefile now defaults to sm_120 and a current GCC/CUDA toolchain.

### Fixes in v1.2

* A mistyped `1...` address is rejected (wrong Base58 checksum or length) instead of being searched
  for forever without a match.
* `-backup` refuses a file written by a build with another group size (`VS_GRP`), whose progress
  would map to the wrong keys and skip part of the range.
* With `-stop` and several addresses, a pause (`p`) no longer forgets the addresses already found,
  so the search still stops once all of them are found.
* An address is marked as found only after its key is verified.
* A very long line in the `-i` file no longer overflows a stack buffer in the Base58 decoder.
* `-random` draws its jumps from a 64-bit generator seeded with 256 bits instead of a 32-bit seed.
* `make` alone builds the program, and a header change rebuilds what depends on it.
* `VanitySearch.vcxproj` uses the installed CUDA toolkit (`CUDA_PATH`, or `/p:CudaVersion=…`) and
  architectures that CUDA 13 still supports (sm_75 … sm_120).
* `bench/build.py` finds any Visual Studio 2022 edition, and builds older refs with their own
  source list.
* The notebook does not record the same found key again in every session.
* Unused code was removed: leftover RCKangaroo headers, the wildcard and SSE address checks, P2SH
  branches, empty GPU checks, and the original prebuilt FixedPaul binaries (`VanitySearch 2.2/`,
  still in the git history).

### Fixes in v1.1

* The kernel no longer writes past the hit buffer when one launch finds more than `-m` hits.
  Prefix searches could corrupt GPU memory before.
* Without `-start`, the search began at key 0, and keys at the start of the range were missed.
  It now starts at 1.
* Pause and resume (`p`) freed the GPU buffers twice and raced the GPU thread (use after free).
  In `-random` mode, every key found after a resume was reported wrongly and discarded.
* The backup is written atomically. It records the range and the GPU layout, and resuming on another
  range or GPU model is refused instead of silently skipping keys. Old backup files still load.
* On Linux, the terminal is no longer left without echo, or with a non-blocking stdin, when the
  tool exits on an error or Ctrl+C. Background jobs and piped stdin no longer touch the terminal.
* The arguments are validated: bad hex, missing values and out-of-range numbers now give an error
  instead of a crash or silent garbage.
* Other fixes: the `-random` seed differs per process; an invalid `-gpuId` gives an error instead of
  hanging; Blackwell core counts are shown; the private key is zero-padded on every platform; the
  progress and ETA no longer show `51200%` or negative times; P2SH (`3...`) addresses, which the
  kernel cannot match, are rejected with a message.

## Usage

```
vanitysearch [-gpuId N] [-start HEX] [-range BITS | -end HEX] [-stop] [-random] [-backup]
             [-i file] [-o file] [-m N] address...
```

| Option | |
|---|---|
| `-start HEX` | First private key, default 1. A `0x` prefix is accepted. |
| `-range BITS` | Scan `start` … `start + 2^BITS − 1` (default 30). |
| `-end HEX` | **New:** last key (inclusive), for ranges that are not a power of two. |
| `-stop` | Stop once every address has been found. |
| `-random` | At every step all GPU threads jump to the same random offset within their own slice of the range. |
| `-backup` | Resume from `VSbackup_gpu<N>.dat`. The file is written every 60 launches, on pause, on Ctrl+C and at the end. A new run without `-backup` keeps the previous file as `.prev`. |
| `-i file` / `-o file` | Addresses to search (one per line, `#` comments) / file the found keys are appended to (with public key and time). |
| `-gpuId N` | GPU to use. One GPU per process: run one instance per GPU. |
| `-l` | **New:** list the CUDA devices. |
| `-h`, `-v` | Help, version. |

While running, `p` pauses and releases the GPU, and `p` again resumes. **Ctrl+C** saves the progress
and exits (a second Ctrl+C exits at once). The exit code is 1 on errors and 0 otherwise.

## Build

**Linux**, CUDA ≥ 12.8 for sm_120:

```
make                 # sm_120 (Blackwell)
make CCAP=89         # other GPUs: 89 = Ada, 86 = Ampere, ...
```

**Windows** (Visual Studio 2022, any edition with "Desktop development with C++", + CUDA ≥ 12.8 + Python 3):

```
python bench/build.py opt            # -> build/opt/vanitysearch.exe, sm_120 only (--arch 89 for Ada, ...)
```

`VanitySearch.sln` also builds, for sm_75 … sm_120, with the CUDA toolkit that `CUDA_PATH` points
to (`msbuild /p:CudaVersion=12.8` picks another installed one).

## Verify before you trust it

```
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 36 --n 3            # sequential
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 34 --n 3 --random   # -random mode
```

The self-test hides a random key in a small range and runs the tool. It then re-derives the
reported key's address in pure Python, independently of the tool. `--target` selects the GPU path:
`single` (default, one `1...` address), `bech32` (one `bc1q...` address) and `prefix` (the first
characters of the address; any key of the range with that prefix passes).

## Google Colab

[`colab/puzzle71_blackwell.ipynb`](colab/puzzle71_blackwell.ipynb) is a self-contained notebook.
Its UI is in Turkish. It clones this repository (`VS_REPO` / `VS_REF`, which can also be a local
clone), builds it for the attached GPU and runs a self-test before searching. A commit that was already
built with the same compiler settings is reused instead of rebuilt. If the self-test fails, it
falls back to the original FixedPaul source from this repository's first commit. Found keys are checked
independently in Python and saved to Google Drive. It targets Bitcoin puzzle #71 and explains the
realistic odds up front. The same notebook also runs outside Colab, in any Jupyter on Linux (apt-based
CUDA install when root) or Windows (CUDA Toolkit + VS 2022 Build Tools). There, state is kept in
`~/vanitysearch-puzzle/`, or in `$PUZZLE_HOME` if set.

## Credits and license

* Original VanitySearch: Jean-Luc Pons (GPLv3)
* VanitySearch-Bitcrack: FixedPaul (GPLv3)
* Blackwell optimisation, tooling and report: VanitySearch-Blackwell (2026)

Licensed under the GNU General Public License v3, see [LICENSE.txt](LICENSE.txt). Modified files
carry a notice in their header. The full change history is in git.

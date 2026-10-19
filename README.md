# VanitySearch-Blackwell

Based on [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul/VanitySearch-Bitcrack),
which is in turn based on [Jean-Luc Pons' VanitySearch](https://github.com/JeanLucPons/VanitySearch). It is
tuned for NVIDIA Blackwell GPUs (sm_120) through profiling.

The tool searches a private-key range for a compressed P2PKH address. The command line is unchanged
from upstream, see [README.upstream.md](README.upstream.md). The GPU kernel needs **10.8% fewer SM
cycles per key**, and every change was A/B measured and verified.

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

## Build

**Linux**, CUDA ≥ 12.8 for sm_120:

```
make                 # sm_120 (Blackwell)
make CCAP=89         # other GPUs: 89 = Ada, 86 = Ampere, ...
```

**Windows** (Visual Studio 2022 Build Tools + CUDA ≥ 12.8 + Python 3):

```
python bench/build.py opt            # -> build/opt/vanitysearch.exe
```

## Verify before you trust it

```
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 36 --n 3            # sequential
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 34 --n 3 --random   # -random mode
```

The self-test hides a random key in a small range and runs the tool. It then re-derives the
reported key's address in pure Python, independently of the tool.

## Google Colab

[`colab/puzzle71_blackwell.ipynb`](colab/puzzle71_blackwell.ipynb) is a self-contained notebook.
Its UI is in Turkish. It builds upstream VanitySearch-Bitcrack plus this project's GPU changes (embedded
as a patch) for the attached GPU and runs a self-test before searching. Found keys are checked
independently in Python and saved to Google Drive. It targets Bitcoin puzzle #71 and explains the
realistic odds up front.

## Credits and license

* Original VanitySearch: Jean-Luc Pons (GPLv3)
* VanitySearch-Bitcrack: FixedPaul (GPLv3)
* Blackwell optimisation, tooling and report: VanitySearch-Blackwell (2026)

Licensed under the GNU General Public License v3, see [LICENSE.txt](LICENSE.txt). Modified files
carry a notice in their header. The full change history is in git.

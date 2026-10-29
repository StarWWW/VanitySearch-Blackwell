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

### Multi-address search (bloom filter)

Pass several full addresses (on the command line or with `-i file`) and the GPU tests every key against a
bloom filter of their hash160s. The filter uses 32–64 bits per address and 8 probes, so most keys cost
about 1.2 memory reads. The CPU then removes the rare false positives (rate 4·10⁻⁸ to 6·10⁻⁶). One
address still uses the constant-memory fast path, and prefixes still use the lookup tables. Measured on
an RTX 5050 Laptop, a 2^34 range with 5 planted keys hidden among random addresses:

| Addresses | Bloom filter | Old 16/32-bit tables (`-nobloom`) |
|---|---|---|
| 1 | 1173 MK/s (single-target path) | same |
| 1,000,005 | 1143 MK/s (4 MB filter), 5/5 found | 1090 MK/s, 5/5 found |
| 10,000,005 | 1099 MK/s (64 MB filter), 5/5 found | 967 MK/s, **1/5 found** (988,345 hits lost) |

With 10M addresses the old tables pass about 0.2% of all keys to the CPU, so the hit buffer
overflows and real hits are dropped. The bloom filter passes about 72 per launch.

### Address lists from Hugging Face

[`tools/fetch_addresses.py`](tools/fetch_addresses.py) downloads a list of funded Bitcoin addresses
from the [`pymmdrza/Bitcoin_Wallet_Addresses`](https://huggingface.co/datasets/pymmdrza/Bitcoin_Wallet_Addresses)
dataset and writes it as a plain `.txt` for `-i`.

```
python tools/fetch_addresses.py p2pkh   -o rich_p2pkh.txt    # 1... addresses
python tools/fetch_addresses.py bech32  -o rich_bech32.txt   # bc1q... addresses
python tools/fetch_addresses.py --list                       # show the available lists
vanitysearch -start 1 -range 40 -stop -i rich_p2pkh.txt
```

The dataset is **gated**: open its page while signed in and accept the terms, create a read token at
<https://huggingface.co/settings/tokens>, then pass it with `--token` or set `HF_TOKEN`. The script
streams the download straight to the `.txt`, so the full file is never held in memory, but the lists
are large (hundreds of MB compressed, a few GB uncompressed). Search **one address type per run** —
the tool takes the type of the first address and ignores the rest, so use the `p2pkh` file for a
P2PKH search and the `bech32` file for a BECH32 search. The `p2sh` (`3...`) list is not searchable by
this tool's kernel.

> A random search against arbitrary funded wallets has no realistic chance of a hit: their keys are
> uniform over 2^256. This is useful for benchmarking the filter at scale or for weak-key research on
> keys you already control, not for finding other people's keys.

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
| `-random` | Each GPU thread jumps to a random key in its slice at every step. |
| `-backup` | Resume from `VSbackup_gpu<N>.dat`. The file is written every 60 launches, on pause, on Ctrl+C and at the end. A new run without `-backup` keeps the previous file as `.prev`. |
| `-i file` / `-o file` | Addresses to search (one per line, `#` comments) / file the found keys are appended to (with public key and time). |
| `-nobloom` | **New:** use the old 16/32-bit tables instead of the bloom filter for several full addresses. |
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
Its UI is in Turkish. It clones this repository (`VS_REPO` / `VS_REF`, which can also be a local
clone), builds it for the attached GPU and runs a self-test before searching. If the self-test fails, it
falls back to the original FixedPaul source from this repository's first commit. Found keys are checked
independently in Python and saved to Google Drive. It targets Bitcoin puzzle #71 and explains the
realistic odds up front. Set `SEARCH_MODE = "multi"` to search an address list instead of the single
puzzle address: the notebook downloads the chosen list with `tools/fetch_addresses.py` (needs an
`HF_TOKEN`, see above) and searches it through the GPU bloom filter. The same notebook also runs
outside Colab, in any Jupyter on Linux (apt-based
CUDA install when root) or Windows (CUDA Toolkit + VS 2022 Build Tools). There, state is kept in
`~/vanitysearch-puzzle/`, or in `$PUZZLE_HOME` if set.

## Credits and license

* Original VanitySearch: Jean-Luc Pons (GPLv3)
* VanitySearch-Bitcrack: FixedPaul (GPLv3)
* Blackwell optimisation, tooling and report: VanitySearch-Blackwell (2026)

Licensed under the GNU General Public License v3, see [LICENSE.txt](LICENSE.txt). Modified files
carry a notice in their header. The full change history is in git.

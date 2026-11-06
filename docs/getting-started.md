# Getting started

This page takes you from a fresh checkout to a first verified search. It should take a few minutes,
plus the compile time.

## Requirements

* An NVIDIA GPU and its driver (`nvidia-smi` must work). The kernel is tuned for Blackwell (sm_120),
  but it builds and runs on older GPUs too (`--arch` / `CCAP`, see below).
* The CUDA Toolkit. Blackwell needs **CUDA 12.8 or newer**.
* An x86-64 CPU. The CPU side uses SSE code, so ARM machines are not supported.
* Linux: `make` and `g++`.
* Windows: **Visual Studio 2022** (any edition, with the "Desktop development with C++" workload)
  and **Python 3**.

## Build

### Linux

```bash
make
```

This builds `./vanitysearch` for sm_120. For another GPU, pass its compute capability:

```bash
make CCAP=89
```

| `CCAP` | GPUs |
|---|---|
| `120` (default) | RTX 50 series, RTX PRO 6000 Blackwell |
| `89` | RTX 40 series (Ada) |
| `86` | RTX 30 series (Ampere) |

If CUDA is not in `/usr/local/cuda`, point to it with `make CUDA=/usr/local/cuda-12.8`.
`make clean` removes the object files, and `make debug=1` builds with debug information.

### Windows

```bash
python bench/build.py opt
```

The executable is written to `build/opt/vanitysearch.exe`. The script finds Visual Studio 2022 and the
CUDA Toolkit by itself. For another GPU, pass its compute capability:

```bash
python bench/build.py opt --arch 89
```

`VanitySearch.sln` also builds in Visual Studio, for sm_75 … sm_120, with the CUDA Toolkit that
`CUDA_PATH` points to.

## Check that it works

Before you trust a build with a long search, let it find a key that you know is there:

```bash
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 36 --n 3
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 34 --n 3 --random
```

(On Linux the executable is `./vanitysearch`.) Each run hides a random key in a small range, lets
the tool search it, and re-derives the reported key's address in pure Python, independently of the
tool. Every line must say `PASS`:

```
  [1/3] PASS  key=312ec521ca41efa  4.5s
  [2/3] PASS  key=2eb43d3331db18a  6.4s
  [3/3] PASS  key=2cb93973d9a8ede  2.1s
```

## A first search

Bitcoin puzzle #30 was solved long ago, so it makes a good test: its key is between `2^29` and
`2^30 - 1`, a range the GPU scans in under a second.

```bash
vanitysearch -start 20000000 -range 29 -stop 1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps
```

Output, shortened:

```
VanitySearch-Blackwell v1.2 (VanitySearch-Bitcrack 2.2 by FixedPaul, tuned for Blackwell)
[keyspace]  range=2^29.00
[keyspace]  start=20000000
[keyspace]    end=3FFFFFFF
Search: 1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps [Compressed]
GPU: GPU #0 NVIDIA GeForce RTX 5050 Laptop GPU (20x128 cores) Grid(4096x128)
Starting keys set in 0.08 seconds

Public Addr: 1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M8diLSC5MyERoW
Priv (HEX): 0x000000000000000000000000000000000000000000000000000000003D94CD64
Pub  Key  : 030D282CF2FF536D2C42F105D0B8588821A915DC3F9A05BD98BB23AF67A2E92A5B

Range Finished! - Average Speed: 1125.0 [MK/s] - Found: 1
Found 1 key(s)
```

Here `-start` and `-range` select the keys `0x20000000 … 0x3FFFFFFF`, and `-stop` ends the search once
the address is found. Every option is described in [usage.md](usage.md).

## Next steps

* [usage.md](usage.md): all options, address formats, ranges, random mode, pause, backup and resume.
* [puzzle-search.md](puzzle-search.md): setting up a search for a puzzle range, and what to expect.
* [troubleshooting.md](troubleshooting.md): error messages and what to do about them.
* [how-it-works.md](how-it-works.md): what the GPU and the CPU do for every key.

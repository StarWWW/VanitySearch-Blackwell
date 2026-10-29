# Optimising VanitySearch-Bitcrack for NVIDIA Blackwell (sm_120)

This report describes the changes in VanitySearch-Blackwell, how they were measured, and the
experiments that did not work and why. Every number below was measured. Where a
value is an estimate, the text says so.

## TL;DR

| | Original | VanitySearch-Blackwell |
|---|---|---|
| SM cycles per key (sm_120, locked clocks) | 1.000 | **0.892 (−10.8%)** |
| RTX 5050 Laptop, 80 W power cap, wall clock | 1070 MK/s | **1166 MK/s (+9.0%)** |
| RTX PRO 6000 Blackwell (Colab), wall clock | 8625 MK/s | **9479 MK/s (+9.9%)**¹ |

¹ Same card, measured in separate Colab sessions. The intermediate version (first
three changes below) ran at 9015 MK/s. The final version adds +5.1% on top of it,
and the cycle measurement predicted +4.9%. The total of +9.9% is below the ≈ +12%
the cycle count implies. Part of the gap is the power limit: with the final
version the card reaches 598 W of its 600 W limit and its clock drops by up to
~1% (see below).

## Two metrics, because GPUs hit different limits

* **Wall-clock A/B** (`bench/bench.py speed`). The builds run interleaved
  (A B A B …) on the same range and the result is the median of the rounds.
  Noise is about ±0.5%. The development GPU was an RTX 5050 Laptop. It
  constantly sits at its **80 W power cap** (`clocks_throttle_reasons = 0x4`),
  so a change that adds instructions also lowers the clock and can look neutral
  even when it saves cycles.
* **SM cycles per key** (`bench/cycles.py`). Nsight Compute measures one
  `comp_keys` launch with clocks locked to base and normalises by the work per
  launch. This is the right metric for a **clock-limited** GPU. The RTX PRO 6000
  draws 506 W of its 600 W limit at its 2430 MHz maximum clock, so on that card
  a saved cycle is a faster key.

The first three changes were chosen by wall clock and the last two by cycles. Two
of those wins had first been rejected because of the laptop's power cap.

## Changes shipped

| Change | Wall clock (laptop) | Cycles / key |
|---|---|---|
| Single full-address search compares hash160 against constant memory instead of the 128 KB prefix table | +0.8% | |
| `_GetHash160Comp` and `CheckPoint` inlined (`HASH_INLINE`, `CHECK_INLINE`) | +1.1% | |
| 512→320-bit reduction uses real multiplies instead of the shift/add expansion of ×0x1000003D1 (`VS_REDMUL`) | +5.0% | |
| *Subtotal* | *1074 → 1154 MK/s, +7.4%* | *−6.35%* |
| SHA-256 / RIPEMD-160 additions written as `IMAD x·1 + y` (`VS_FADD=7`) | ≈ 0 (clock drops at 80 W) | −2.1% |
| 128 threads per block, twice the blocks (`VS_TPB=128`, `VS_BPSM=256`; same threads per launch) | | −2.1% |
| **Total** | **1070 → 1166 MK/s, +9.0%** | **−10.8%** |

Also included, measured neutral: removing the `"memory"` clobbers from the
arithmetic inline asm (ptxas already emitted the same code).

Correctness: `bench/bench.py selftest` hides a random key in an aligned range,
runs the tool, and re-derives the reported key's address in pure Python. Every
shipped change passed it in sequential mode and in `-random` mode.

## Why the kernel is where it is

Nsight Compute on the shipped kernel, before the last two changes:

* **Pipe-bound, not memory-bound.** ALU-heavy pipe 79% busy, FMA pipe 34%,
  memory 7%. About 3340 SASS instructions per key.
* **Ablation.** Dropping RIPEMD-160 gives +23% speed; dropping the whole hash
  gives +118%. Time splits roughly 54% hashing and 46% elliptic-curve
  arithmetic plus checks.
* **Stall sampling, per instruction.** Math-pipe throttle 33%, not selected 28%,
  fixed-latency wait 16%, local memory 1.9%, barrier 1.5%. Latency hiding and
  memory have no headroom left.
* **sm_120 pipe map**, from `bench/pipes2.cu` plus Nsight Compute pipe counters:

  | Instructions | Pipe | Warp-instructions / SM / clock |
  |---|---|---|
  | SHF, LOP3, **IADD3**, LEA | ALU-heavy only | 2 |
  | IADD (2-input), IADD.X, ISETP/SEL | ALU-lite: split between ALU-heavy and the FMA pipe's alulite sub-pipe | 4 |
  | IMAD | FMA-heavy only | 2 |

  SHA-256 and RIPEMD-160 consist almost entirely of rotations (SHF/LEA) and
  boolean functions (LOP3). That work is fixed by the algorithms and must run on
  the ALU-heavy pipe. The wins above came from moving the *movable* work
  (multiplications in the reduction, 3-input additions) to the under-used FMA
  pipe.
* **Elliptic-curve side.** It is already at the minimum of the affine
  batch-inversion method: 3.5 multiplications + 1 squaring per key, with one
  inversion shared by 1024 keys.
* **Host side.** Nsight Systems shows the GPU busy 99.8% of the time. The
  ~1.2 ms between ~590 ms kernels leaves at most 0.2% to gain.

On the RTX PRO 6000 the final version draws 598 W of 600 W at 2392–2422 MHz. The
card is now at both its clock and power limits. Further gains would need fewer
instructions per key, not a different instruction mix.

## Experiments that did not work

| Experiment | Result |
|---|---|
| Remove `__syncthreads()` (no shared memory is used) — `VS_NOBARRIER` | laptop −1.0%; cycles −2.4% with 256-thread blocks, but it does not stack with 128-thread blocks (−2.8% vs −4.2%) |
| Group size 512 / 256 (`VS_GRP`) | laptop −0.8% / −2.0%; cycles per key +0.6% (512) |
| 512 threads/block; grid SM×64; `maxrregcount` 96 / 80; `__launch_bounds__(…, 2)` | cycles per key +4.8%, +1.1%, +2.6% / +1.4%, +0.5% |
| Rotations on the FMA pipe via `IMAD.WIDE` (x·2ⁿ, halves combined with OR) | laptop −8 … −27% (the OR is a LOP3 on the ALU pipe again) |
| Same, halves combined with an add (ALU instructions really drop, LEA.HI 648 → 0) | laptop −3.9 … −17.6%, cycles +3.1% (IMAD.WIDE is half-rate and long-latency, and hash rounds are a serial chain) |
| 64-bit carry chains as 32-bit IADD/IADD.X pairs | laptop ±0, cycles +0.2% |
| 256×256 product with 32-bit limbs and PTX carry chains (`bench/fieldmul.cu`) | bit-exact on 65536 vectors and +2% in isolation, but **+3.6% cycles** in the kernel |
| Full grid (SMs × 256 blocks) instead of rounding the sequential grid down to a power of two | cycles per key −0.11% on the RTX 5050 (5120 vs 4096 blocks): the tail of a launch is negligible, and the power of two keeps an exact split of power-of-two ranges and the backup layout |
| Profiling read-out "long scoreboard 18%, barrier 14%" | wrong metric units; the per-instruction sampling above is the reliable one |

## Reproducing (Windows; Linux uses the Makefile)

```
python bench/build.py base --ref ":/^Import VanitySearch-Bitcrack"   # original: the first commit is the unmodified upstream import
python bench/build.py opt                             # this tree
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 36 --n 3
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 34 --n 3 --random
python bench/bench.py speed  build/base/vanitysearch.exe build/opt/vanitysearch.exe
python bench/cycles.py       build/base/vanitysearch.exe build/opt/vanitysearch.exe
```

`bench/cycles.py` needs Nsight Compute and GPU performance-counter access. A
build with a different amount of work per launch takes a suffix, e.g.
`build/x/vanitysearch.exe@0.5`.

Build-time options (`-D…`, see `bench/sweep.py`):

| Option | Default | Meaning |
|---|---|---|
| `VS_REDMUL` | 1 | multiply-based 512→320 reduction |
| `VS_FADD` | 7 | IMAD additions: 1 = SHA-256 rounds, 2 = SHA-256 schedule, 4 = RIPEMD-160 |
| `VS_TPB` / `VS_BPSM` | 128 / 256 | threads per block / blocks per SM per launch |
| `VS_GRP` | 1024 | keys per thread per launch (256, 512 or 1024) |
| `VS_NOBARRIER` | off | drop the block barriers |
| `VS_MINBLOCKS` | unset | `__launch_bounds__` minimum blocks |
| `HASH_INLINE`, `CHECK_INLINE` | `__forceinline__` | inlining of the hash and check functions |

Microbenchmarks: `bench/pipes.cu` (per-instruction throughput), `bench/pipes2.cu`
(pipe sharing; read its Nsight pipe counters rather than its own timing), and
`bench/fieldmul.cu` (field-multiply variants).

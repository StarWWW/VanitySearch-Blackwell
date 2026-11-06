# How it works

A short tour of what happens for every key. For the Blackwell tuning and its measurements, see
[OPTIMIZATION.md](../OPTIMIZATION.md).

## The problem

A Bitcoin address is derived from a private key by a fixed pipeline:

```
private key  --[secp256k1 scalar mult]-->  public key
             --[SHA-256 then RIPEMD-160]-->  hash160 (20 bytes)
             --[Base58Check or bech32]-->    address
```

The tool runs this pipeline for every key in a range and checks the resulting `hash160` against the
target(s). There is no shortcut through it; the work is to run it as fast as the GPU allows and to
check the result cheaply.

## One group of keys at a time

The slow step is the elliptic-curve scalar multiplication. Doing it independently per key would be
far too slow, so the tool uses the standard VanitySearch trick:

* Each GPU thread owns a contiguous slice of the range. Before the first launch, the host computes
  every thread's starting point once (by point additions with a batch inversion, not one scalar
  multiplication per thread) and copies them to the GPU.
* In each launch, a thread holds one point `P` at the middle of its current group and reaches the
  other keys of the group as `P ± i·G` with elliptic-curve **point additions**, which are much cheaper
  than a multiplication. It then moves `P` forward by one group for the next launch.
* Point addition needs a modular inverse, and inverses are expensive. They are done in **batch**: the
  group's denominators are multiplied together, inverted once, and unwound. The cost comes out to
  about 3.5 field multiplications and one squaring per key, with no per-key inverse.

A group is `VS_GRP` keys per thread per launch (1024 by default). All the arithmetic is in affine
coordinates (Z = 1).

## Hashing and checking on the GPU

For each public key the kernel computes `RIPEMD-160(SHA-256(compressed pubkey))` and then checks the
20-byte `hash160`:

* **One full address** — the fast path. The `hash160` is held in constant memory and compared
  directly, one comparison per key.
* **Several full addresses** — a 16-bit prefix of the `hash160` indexes a table; on a prefix hit a
  second table of 32-bit values is searched. Most keys are rejected by the first table with a single
  load.
* **A prefix** (leading address characters) — handled through the same tables.

A key that passes the GPU check is written to a small output buffer with its thread id and offset, so
the host can reconstruct the exact private key. The buffer has a fixed capacity (`-m`); if a single
launch somehow overflows it, the extra hits are **counted but not written** (you get a "items lost"
warning) rather than corrupting memory.

## The host confirms every hit

The tables compare only 16 or 32 bits of the hash160, so the GPU also reports hashes that merely share
those bits with a target. So before anything is reported, the CPU:

1. rebuilds the private key from the thread id and offset,
2. recomputes the public key and address in its own code,
3. compares that address to the target.

Only an exact match is printed and saved. This is also what the self-test relies on: it re-derives the
reported key's address in **pure Python**, independently of the tool, so a miscompiled build cannot
fool it. (If a build's own CPU SHA-256 is miscompiled, it still prints the candidate key as `CandKey:`
so an external check can use it, and warns at start-up.)

## Sequential vs random

* **Sequential** walks each thread's slice in order. Progress is well defined, so it can be saved and
  resumed (`-backup`), and the percentage and ETA are meaningful.
* **Random** (`-random`) makes every thread jump to the same random offset in its slice each step.
  Over a range too large to finish, this samples it uniformly; there is no progress to save. The
  offsets come from a 64-bit generator seeded with 256 bits of entropy, so separate processes and
  restarts do not repeat each other.

## Why Blackwell specifically

The kernel is limited by the GPU's ALU pipe, which is kept busy by the rotations and boolean functions
of SHA-256 and RIPEMD-160. The FMA (multiply) pipe sits mostly idle. The Blackwell tuning moves the
work that *can* move — the reduction multiplies and the three-input additions — onto that idle pipe,
expressing additions as `IMAD x·1 + y`, and reshapes the launch (128 threads per block). Net result:
**10.8% fewer SM cycles per key**. The full story, including the ideas that did not pay off, is in
[OPTIMIZATION.md](../OPTIMIZATION.md).

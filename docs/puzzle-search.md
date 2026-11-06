# Searching a Bitcoin puzzle

The [Bitcoin puzzle](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx) is a set of addresses whose
private keys lie in known ranges: puzzle **N** has its key between `2^(N-1)` and `2^N - 1`. The low
puzzles, and many higher ones, have been solved. The unsolved ones still carry a prize; check a puzzle
list for which ones are still open. This page shows how to point the tool at one and what to expect.

## The range of a puzzle

Puzzle `N` ⟶ `-start` is `2^(N-1)` and the range is `N-1` bits:

| Puzzle | `-start` (hex) | `-range` |
|---|---|---|
| 66 (solved) | `20000000000000000` | `65` |
| 71 | `400000000000000000` | `70` |
| 72 | `800000000000000000` | `71` |

In general, for puzzle `N`: `-start` is `1` followed by `(N-1)` zero **bits** (so a `1` then `(N-1)/4`
hex digits when `N-1` is a multiple of 4), and `-range` is `N-1`.

## Running it

A puzzle range is far too large to scan to the end, so use random mode:

```bash
vanitysearch -gpuId 0 -random -stop -o found.txt -start 400000000000000000 -range 70 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
```

* `-random` samples the range instead of walking it (see [usage.md](usage.md#random-mode)).
* `-stop` ends the search once the address is found. Without it, the tool reports the key and keeps
  searching.
* `-o found.txt` makes sure a hit is written to disk, not only to the screen.

If you would rather scan a chosen sub-range in order and be able to stop and resume it, drop `-random`
and use a smaller `-range` with `-backup`:

```bash
vanitysearch -gpuId 0 -start 400000000000000000 -range 40 -backup -stop -o found.txt 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
```

## What to expect

Read this before you spend money or electricity on it.

Puzzle 71's range holds `2^70 ≈ 1.2 × 10^21` keys. There is no public key on the chain for an
unspent puzzle address, so the fast methods (Kangaroo, BSGS) do not apply — brute force is the only
way. At ~10 Gkey/s, one strong GPU would need on the order of **thousands of years** to cover the
whole range.

A random search does not "get closer" over time: every key is an independent draw, so your chance in
one day is just (keys tried that day) / (range size). For puzzle 71 at 10 Gkey/s that is about
1 in 1.4 million per day. Running more GPUs multiplies your keys per day and your chance by the same
factor, and nothing else.

**Treat this as a lottery, not an investment.** It is a fun way to put a GPU to work and to learn how
the search works, but the expected value is far below the cost of the electricity. Only spend compute
you are happy to lose.

A balance check before a long run avoids wasting it on an address whose prize is already gone — see
[privatekeys.pw](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx) or any block explorer.

## If you ever do find a key

A puzzle address that has never been spent has no public key on the chain — that is what keeps it out
of reach of Kangaroo/BSGS. **The moment you spend from it, the public key becomes visible.** For a key
known to sit in a small range, bots can then derive it in seconds and try to replace your transaction
with their own (higher fee) to steal the prize; this has happened on earlier puzzles.

So, if you find one:

1. Save `found.txt` somewhere offline first (write it down, copy it to an encrypted drive).
2. Build and sign the spending transaction offline.
3. Send it to a miner through a private channel (a service that mines a transaction without putting it
   in the public mempool), so the public key is not exposed before the transaction is mined.

This is general information, not financial or security advice.

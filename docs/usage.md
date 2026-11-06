# Usage

```
vanitysearch [-gpuId N] [-start HEX] [-range BITS | -end HEX] [-stop] [-random] [-backup]
             [-i file] [-o file] [-m N] [-l] [-h] [-v] address...
```

On Linux the program is `./vanitysearch`; on Windows it is `build\opt\vanitysearch.exe`. Give at least
one address (or `-i file`). With no arguments the tool prints its help and exits.

## Options

| Option | Meaning |
|---|---|
| `-start HEX` | First private key of the range, in hex (a `0x` prefix is accepted). Default `1`. |
| `-range BITS` | Range size: scan `start` … `start + 2^BITS - 1`. `BITS` is 1…256, default 30. |
| `-end HEX` | Last private key of the range (inclusive), instead of `-range`. For ranges that are not a power of two. |
| `-stop` | Stop as soon as every given address has been found. Without it, the search reports each hit and keeps going. |
| `-random` | Random mode (see below). Without it the range is scanned in order. |
| `-backup` | Resume a sequential search from its backup file (see [Backup and resume](#backup-and-resume)). |
| `-i file` | Read the addresses from a file, one per line. Blank lines and lines starting with `#` are ignored. |
| `-o file` | Append every found key to this file as well as printing it. |
| `-m N` | Maximum hits stored per kernel launch, default 262144. Raise it only if you search a huge list and see an "items lost" warning. |
| `-gpuId N` | Which GPU to use (default 0). See [Several GPUs](#several-gpus). |
| `-l` | List the CUDA devices with their index, name and memory, then exit. |
| `-h` | Print the built-in help and exit. |
| `-v` | Print the version and exit. |

A range given two ways is refused: use `-range` or `-end`, not both. A key beyond the secp256k1 curve
order is clipped to it, and `-start 0` becomes `-start 1` (0 is not a valid private key).

## Addresses

The GPU hashes the **compressed** public key of each candidate, so it matches:

* **P2PKH**, starting with `1` (e.g. `1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps`).
* **P2WPKH (bech32)**, starting with `bc1q` (e.g. `bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4`).

All the addresses of one run must be the same type; the type of the first address wins and the rest are
ignored with a message. **P2SH** (`3...`) addresses cannot be matched and are rejected.

You can also give a **prefix** instead of a full address: the leading characters of an address, such
as `1Love` or `bc1qxyz`. The tool then reports every key in the range whose address starts with those
characters. A search for **one** full address is the fastest case, because the GPU compares each
hash160 directly against constant memory.

A mistyped `1...` address (wrong Base58 checksum or wrong length) is reported and skipped instead of
being searched for forever.

## Output

When a key is found the tool prints, and (with `-o`) appends to the file:

```
Public Addr: 1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M8diLSC5MyERoW
Priv (HEX): 0x000000000000000000000000000000000000000000000000000000003D94CD64
Pub  Key  : 030D282CF2FF536D2C42F105D0B8588821A915DC3F9A05BD98BB23AF67A2E92A5B
```

The WIF prefix is `p2pkh:` for `1...` addresses and `p2wpkh:` for `bc1q...` addresses. The key is
printed only after the tool re-derives the address from it and confirms the match.

## The progress line

While it runs the tool prints one status line, updated in place:

```
1185.7 MK/s - 29 BKeys - 2^34.75 [0.00%] - RUN: 00:00:24|END: 3.16e+04 years - Found: 0
```

(An RTX 5050 Laptop on puzzle 71's `2^70` range.)

* **MK/s**: millions of keys per second, a smoothed average.
* **BKeys**: billions of keys scanned so far. A search resumed with `-backup` also counts its earlier
  progress.
* **2^x**: the same count as a power of two.
* **[%]**: the share of the range scanned. For a huge range it stays at 0.00%.
* **RUN**: elapsed time, including the earlier run time of a resumed search.
* **END** (sequential mode only): the estimated time left to scan the whole range, as `HH:MM:SS`,
  `Nd HH:MM:SS`, a number of years when it is longer than 100 years, or `--:--:--` when it cannot be
  estimated.
* **Found**: keys found so far.

## Controls

* **`p`** pauses the search and **releases the GPU**, so another program can use it. Press `p` again
  to resume. In sequential mode, pausing writes a backup first.
* **Ctrl+C** saves the progress and exits. A second Ctrl+C exits immediately.

These work only when the tool runs in a real terminal. With input redirected (a pipe or `</dev/null`)
the terminal is left untouched and there are no keyboard controls.

## Random mode

With `-random`, at each step every GPU thread jumps to the same random offset inside its own slice of
the range, instead of walking the range in order. Over a huge range (a puzzle, say) you will never scan
the whole thing, so where you start does not matter and sampling at random is as good as any order; it
also means two machines on the same range are unlikely to retread the same keys. `-random` keeps no
progress and cannot be resumed, so `-backup` is not allowed with it.

## Backup and resume

A **sequential** search (without `-random`) writes its progress to `VSbackup_gpu<N>.dat` in the
current directory, where `<N>` is the GPU id. The file is written every 60 kernel launches, when you
pause, on Ctrl+C, when `-stop` ends the search and when the range finishes. It is written atomically,
so an interrupted write cannot corrupt it. Resume from the same directory.

Resume with `-backup`, using the **same** `-start`/`-range` (or `-end`) as the original run:

```bash
vanitysearch -start 8000000000 -range 40 -backup -o found.txt 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
```

If the backup belongs to another range, or was made on a GPU with a different thread count or a build
with a different group size (`VS_GRP`), resuming is refused rather than silently skipping keys. A very
old backup from upstream (12 bytes, no range recorded) still loads, with a warning to check the range
yourself.

When you start a fresh sequential run and a backup already exists, it is moved aside to
`VSbackup_gpu<N>.dat.prev` so you can still resume the earlier search.

## Several GPUs

One process drives one GPU. To use several GPUs, run one instance per GPU, each with its own `-gpuId`
and, if you want them on different parts of the space, its own range. A `-gpuId 0,1` style list is not
supported; only the first id is used, with a warning.

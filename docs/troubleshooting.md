# Troubleshooting

## Building

**`nvcc: command not found` / CUDA not found.** Install the CUDA Toolkit (≥ 12.8 for Blackwell). On
Linux, point the Makefile at it with `make CUDA=/usr/local/cuda-12.8`. On Windows, `bench/build.py`
searches the standard install location and `CUDA_PATH`.

**`Unsupported gpu architecture 'compute_120'`.** Your CUDA Toolkit is older than 12.8. Blackwell
(sm_120) needs 12.8 or newer. For an older GPU, build for its architecture instead, e.g.
`make CCAP=89` or `python bench/build.py opt --arch 89`.

**Windows: the build script cannot find Visual Studio.** Install **Visual Studio 2022** with the
"Desktop development with C++" workload (the Build Tools edition is enough).

## Running

**`invalid GPU id N` / `GPU #N initialisation failed`.** There is no GPU with that index. Run with
`-l` to list the GPUs and their indices, then pass a valid `-gpuId`.

```
GPUEngine: invalid GPU id 9 (1 device(s) found, use -l to list them)
[ERROR] GPU #9 initialisation failed
```

**`There are no available device(s) that support CUDA`.** The driver does not see a CUDA GPU. Check
that `nvidia-smi` works. Under WSL or in a container, make sure the GPU is passed through.

**`Ignoring address "..." (wrong checksum, mistyped address?)`.** The address failed its Base58
checksum — almost always a typo. Copy it again.

**`Ignoring address "..." (P2SH addresses are not supported, use 1... or bc1q...)`.** `3...` addresses
hash a script, not a public key, so this tool cannot match them. Only `1...` and `bc1q...` are
supported.

**`[ERROR] VanitySearch: nothing to search !`** Every address you gave was ignored (wrong type,
checksum or format — look for the `Ignoring address` lines just above). Give a valid `1...` or
`bc1q...` address or prefix.

**`[ERROR] Invalid -start argument, hex digits expected`** (and similar). An option got a bad value.
Keys are hexadecimal (`-start`, `-end`); `-range`, `-m` and `-gpuId` are decimal numbers. `-h` lists
every option.

**`[ERROR] Use either -range or -end, not both`.** Pick one way to set the end of the range.

**`[ERROR] -backup resumes a sequential search, it cannot be used with -random`.** Random mode keeps
no progress, so there is nothing to resume. Drop one of the two flags.

**`[ERROR] ...backup... belongs to another range` / `was made with N GPU threads` / `N keys per
thread`.** The backup does not match this run. Resume with the exact same `-start`/`-range` (or
`-end`), on the same GPU model, with the same build — or start fresh without `-backup` (your old
backup is kept as `VSbackup_gpu<N>.dat.prev`).

**`Warning, N items lost`.** One kernel launch found more hits than the output buffer holds — only
with a very large address list or a very short prefix. Raise the per-launch limit with `-m` (e.g.
`-m 1048576`), or search a longer prefix / fewer addresses.

**The keyboard (`p`, Ctrl+C) does nothing.** Those controls need a real terminal. If you redirected
input (a pipe, `</dev/null`, some launchers), the tool runs without keyboard controls by design.

## Results

**A line says `Warning, wrong private key generated !`.** The CPU re-derivation of a GPU hit did not
match. The key is not reported. On most builds this never happens; if a build prints it for every hit,
its CPU SHA-256 is miscompiled (you will also see a `[WARNING] The CPU SHA-256 of this build is wrong`
line at start-up) — rebuild with the provided scripts / a standard compiler.

**I expected a hit and got none.** Make sure the key is really inside `-start … -start + 2^range - 1`.
For a puzzle, check the range in [puzzle-search.md](puzzle-search.md). A sanity check is to search a
**solved** low puzzle (see [getting-started.md](getting-started.md)); if that is found, the build is
fine and the key you want is simply not in the range you scanned.

## Still stuck?

Run the self-test and note which step fails:

```bash
python bench/bench.py selftest build/opt/vanitysearch.exe --bits 36 --n 3
```

Open an issue with your GPU model, CUDA version, OS, the exact command line and the output.

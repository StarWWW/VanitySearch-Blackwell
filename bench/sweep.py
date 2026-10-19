"""Builds a set of compile-time variants of the working tree and benchmarks them against a reference.

  python bench/sweep.py ref=build/p3/vanitysearch.exe "inl=-DHASH_INLINE=__forceinline__" "tpb128=-DVS_TPB=128"
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def build(tag, flags):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'build.py'), tag, '--ptxas-v', f'--nvflags={flags}'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    regs = [l.strip() for l in r.stdout.splitlines() if 'Used' in l][:1]
    print(f'{tag:>12}: {"ok" if r.returncode == 0 else "BUILD FAILED"}  {regs[0] if regs else ""}', flush=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:])
    return r.returncode == 0


def main():
    ref, variants = None, []
    for a in sys.argv[1:]:
        tag, flags = a.split('=', 1)
        if tag == 'ref': ref = flags
        else: variants.append((tag, flags))
    with ThreadPoolExecutor(3) as ex:
        ok = list(ex.map(lambda v: build(*v), variants))
    exes = ([ref] if ref else []) + [os.path.join(ROOT, 'build', t, 'vanitysearch.exe') for (t, _), k in zip(variants, ok) if k]
    extra = os.environ.get('SWEEP_ARGS', '--seconds 30 --warmup 12 --rounds 2').split()
    subprocess.run([sys.executable, os.path.join(HERE, 'bench.py'), 'speed'] + exes + extra)


if __name__ == '__main__':
    main()

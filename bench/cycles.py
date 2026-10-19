"""Clock-independent comparison: SM cycles of one comp_keys launch, measured by Nsight Compute
with locked base clocks. Use this for clock-limited GPUs (e.g. RTX PRO 6000 at 506 W / 600 W),
where fewer cycles per launch translate 1:1 into speed; the wall-clock A/B in bench.py also
captures power/clock effects (power-capped laptops).

  python bench/cycles.py build/final/vanitysearch.exe build/f7/vanitysearch.exe [--repeat 2]
  python bench/cycles.py build/p3/vanitysearch.exe build/tpb128/vanitysearch.exe@0.5
    (@k = keys per launch relative to the first build, e.g. a half-size grid; results are per key)
"""
import argparse, os, re, statistics, subprocess

NCU = r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.0\ncu.bat'


def cycles(exe):
    r = subprocess.run([NCU, '--kernel-name', 'comp_keys', '--launch-skip', '2', '--launch-count', '1',
                        '--clock-control', 'base', '--metrics', 'sm__cycles_elapsed.avg,sm__inst_executed.sum',
                        exe, '-gpuId', '0', '-start', '400000000000000000', '-range', '35',
                        '1BitcoinEaterAddressDontSendf59kuE'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace',
                       cwd=os.path.dirname(exe), timeout=600)
    out = r.stdout.replace('\r', '\n')
    num = lambda name: float(re.search(name + r'\s+\S+\s+([\d.,]+)', out).group(1).replace('.', '').replace(',', '.'))
    return num('sm__cycles_elapsed.avg'), num('sm__inst_executed.sum')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exes', nargs='+')
    ap.add_argument('--repeat', type=int, default=1)
    a = ap.parse_args()
    res, work = {}, {}
    for _ in range(a.repeat):
        for spec in a.exes:
            e, _, k = spec.partition('@')
            e = os.path.abspath(e)
            work[e] = float(k or 1)
            c, i = cycles(e)
            c /= work[e]
            res.setdefault(e, []).append((c, i))
            print(f'  {c / 1e6:9.2f} Mcycles  {i / 1e9:6.2f} Ginst  {e}', flush=True)
    base = statistics.median(c for c, _ in res[os.path.abspath(a.exes[0].partition('@')[0])])
    print('\ncycles per key vs first (negative = faster on a clock-limited GPU)')
    for e, v in res.items():
        c = statistics.median(x for x, _ in v)
        print(f'  {100 * (c / base - 1):+6.2f}%   {e}')


if __name__ == '__main__':
    main()

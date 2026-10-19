"""Builds vanitysearch.exe on Windows with nvcc + MSVC for one GPU arch.

  python bench/build.py opt                      # current working tree  -> build/opt/vanitysearch.exe
  python bench/build.py base --ref master        # a git ref (exported)  -> build/base/vanitysearch.exe
  python bench/build.py test --nvflags "-maxrregcount=128"

Pass --ptxas-v to print register/local-memory usage of the kernel.
"""
import argparse, glob, os, shutil, subprocess, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VCVARS = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
SRC = ['Base58.cpp', 'IntGroup.cpp', 'main.cpp', 'Random.cpp', 'Timer.cpp', 'Int.cpp', 'IntMod.cpp', 'Point.cpp',
       'SECP256K1.cpp', 'Vanity.cpp', 'GPU/GPUGenerate.cpp', 'hash/ripemd160.cpp', 'hash/sha256.cpp',
       'hash/sha512.cpp', 'hash/ripemd160_sse.cpp', 'hash/sha256_sse.cpp', 'Bech32.cpp', 'Wildcard.cpp',
       'GPU/GPUEngine.cu']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--ref', help='build this git ref instead of the working tree')
    ap.add_argument('--arch', default='120')
    ap.add_argument('--nvflags', default='')
    ap.add_argument('--ptxas-v', action='store_true')
    a = ap.parse_args()

    out = os.path.join(ROOT, 'build', a.tag)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    src = ROOT
    if a.ref:
        src = os.path.join(out, 'src')
        os.makedirs(src)
        zpath = os.path.join(out, 'src.zip')
        subprocess.run(['git', '-C', ROOT, 'archive', '--format=zip', '-o', zpath, a.ref], check=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(src)

    cuda = os.environ.get('CUDA_PATH') or sorted(glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*'))[-1]
    nvcc = os.path.join(cuda, 'bin', 'nvcc.exe')
    flags = (f'-O3 -m64 -gencode arch=compute_{a.arch},code=sm_{a.arch} --threads 0 '
             f'-DWIN32 -DWIN64 -DNDEBUG -D_CRT_SECURE_NO_WARNINGS -D_CONSOLE '
             f'-Xcompiler "/O2 /EHsc /openmp /W0" -I"{src}" -lcudart_static -ladvapi32 {a.nvflags}')
    if a.ptxas_v:
        flags += ' -Xptxas -v'
    files = ' '.join(f'"{os.path.join(src, f)}"' for f in SRC)
    bat = os.path.join(out, 'build.bat')
    with open(bat, 'w') as f:
        f.write(f'@echo off\ncall "{VCVARS}" >nul\ncd /d "{out}"\n'
                f'"{nvcc}" {flags} -o vanitysearch.exe {files}\n')
    r = subprocess.run(['cmd', '/c', bat], capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = r.stdout + r.stderr
    keep = [l for l in log.splitlines() if any(s in l for s in ('error', 'ptxas info', 'bytes stack', 'Used'))]
    print('\n'.join(keep) if keep else log[-3000:])
    exe = os.path.join(out, 'vanitysearch.exe')
    if r.returncode or not os.path.exists(exe):
        print(log[-6000:])
        sys.exit(f'build failed (rc={r.returncode})')
    print('built', exe)


if __name__ == '__main__':
    main()

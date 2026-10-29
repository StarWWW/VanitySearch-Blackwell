"""Correctness self-test and A/B speed benchmark for vanitysearch builds.

  python bench/bench.py selftest build/base/vanitysearch.exe [--bits 36] [--n 3] [--target single]
  python bench/bench.py speed build/base/vanitysearch.exe build/opt/vanitysearch.exe [--seconds 40] [--rounds 3]

selftest hides a random key in an aligned 2^bits range, asks the tool to find it and
verifies the reported key independently (pure-Python secp256k1 + hash160). --target picks the
GPU path under test:
  single  one P2PKH address (constant-memory compare)       bech32  one bc1q address
  bloom   the address among --decoys random ones (-i file)   tables  the same with -nobloom
  prefix  the first characters of the address; any key of the range with that prefix passes
speed runs the builds interleaved (A B A B ...) on the same range so thermal drift
hits all of them equally, and reports the median MK/s after a warm-up period.
"""
import argparse, hashlib, os, re, secrets, statistics, subprocess, sys, threading, time

_P = 2**256 - 2**32 - 977
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
      0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)


def _ec_add(a, b):
    if a is None: return b
    if b is None: return a
    if a[0] == b[0]:
        if (a[1] + b[1]) % _P == 0: return None
        l = 3 * a[0] * a[0] * pow(2 * a[1], -1, _P) % _P
    else:
        l = (b[1] - a[1]) * pow(b[0] - a[0], -1, _P) % _P
    x = (l * l - a[0] - b[0]) % _P
    return (x, (l * (a[0] - x) - a[1]) % _P)


def pubkey_compressed(k):
    r, q = None, _G
    while k:
        if k & 1: r = _ec_add(r, q)
        q = _ec_add(q, q); k >>= 1
    return bytes([2 + (r[1] & 1)]) + r[0].to_bytes(32, 'big')


def _rmd160_py(msg):
    rol = lambda x, n: ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF
    F = [lambda x, y, z: x ^ y ^ z, lambda x, y, z: (x & y) | (~x & z), lambda x, y, z: (x | ~y) ^ z,
         lambda x, y, z: (x & z) | (y & ~z), lambda x, y, z: x ^ (y | ~z)]
    KL = [0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E]
    KR = [0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000]
    RL = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
          3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
          4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13]
    RR = [5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
          15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
          12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11]
    SL = [11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
          11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
          9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6]
    SR = [8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
          9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
          8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11]
    h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
    msg = bytes(msg)
    ml = len(msg) * 8
    msg += b'\x80' + b'\x00' * ((55 - len(msg)) % 64) + ml.to_bytes(8, 'little')
    for off in range(0, len(msg), 64):
        X = [int.from_bytes(msg[off + 4 * i: off + 4 * i + 4], 'little') for i in range(16)]
        al, bl, cl, dl, el = h
        ar, br, cr, dr, er = h
        for j in range(80):
            r = j // 16
            t = (rol((al + F[r](bl, cl, dl) + X[RL[j]] + KL[r]) & 0xFFFFFFFF, SL[j]) + el) & 0xFFFFFFFF
            al, el, dl, cl, bl = el, dl, rol(cl, 10), bl, t
            t = (rol((ar + F[4 - r](br, cr, dr) + X[RR[j]] + KR[r]) & 0xFFFFFFFF, SR[j]) + er) & 0xFFFFFFFF
            ar, er, dr, cr, br = er, dr, rol(cr, 10), br, t
        t = (h[1] + cl + dr) & 0xFFFFFFFF
        h[1] = (h[2] + dl + er) & 0xFFFFFFFF
        h[2] = (h[3] + el + ar) & 0xFFFFFFFF
        h[3] = (h[4] + al + br) & 0xFFFFFFFF
        h[4] = (h[0] + bl + cr) & 0xFFFFFFFF
        h[0] = t
    return b''.join(x.to_bytes(4, 'little') for x in h)


def ripemd160(b):
    try:
        return hashlib.new('ripemd160', b).digest()
    except Exception:
        return _rmd160_py(b)


_B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def b58check(payload):
    b = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n, s = int.from_bytes(b, 'big'), ''
    while n:
        n, r = divmod(n, 58); s = _B58[r] + s
    return '1' * (len(b) - len(b.lstrip(b'\0'))) + s


def hash160_from_key(k):
    return ripemd160(hashlib.sha256(pubkey_compressed(k)).digest())


def addr_from_key(k):
    return b58check(b'\x00' + hash160_from_key(k))


_BECH = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l'


def _bech32_polymod(values):
    gen = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (b >> i) & 1 else 0
    return chk


def p2wpkh(h160):
    """bc1q... address of a hash160 (BIP-173, witness version 0)."""
    acc, bits, data = 0, 0, [0]
    for b in h160:
        acc = acc << 8 | b; bits += 8
        while bits >= 5:
            bits -= 5; data.append(acc >> bits & 31)
    if bits: data.append(acc << (5 - bits) & 31)
    hrp = [ord(c) >> 5 for c in 'bc'] + [0] + [ord(c) & 31 for c in 'bc']
    pm = _bech32_polymod(hrp + data + [0] * 6) ^ 1
    return 'bc1' + ''.join(_BECH[d] for d in data + [pm >> 5 * (5 - i) & 31 for i in range(6)])


assert addr_from_key(1) == '1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH'
assert p2wpkh(hash160_from_key(1)) == 'bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4'

RX_KEY = re.compile(r'Priv \(HEX\):\s*0x\s*([0-9A-Fa-f]+)')
RX_CAND = re.compile(r'CandKey:\s*([0-9A-Fa-f]+)')
RX_SPEED = re.compile(r'([\d.]+)\s*MK/s')


def run_tool(cmd, timeout, on_line):
    """Runs cmd, calls on_line(line) for every \\r/\\n separated line; stops when it returns True."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         cwd=os.path.dirname(os.path.abspath(cmd[0])))
    stop, buf = threading.Event(), b''
    timer = threading.Timer(timeout, lambda: (stop.set(), p.kill()))
    timer.start()
    try:
        while True:
            chunk = p.stdout.read1(65536)
            if not chunk: break
            buf += chunk
            *lines, buf = re.split(rb'[\r\n]+', buf)
            for ln in lines:
                s = ln.decode('utf-8', 'replace').strip()
                if s and on_line(s):
                    stop.set(); break
            if stop.is_set(): break
    finally:
        timer.cancel()
        if p.poll() is None: p.kill()
        p.wait()


def selftest(exe, bits, n, timeout, random=False, target='single', decoys=1000):
    ok_all = True
    to_addr = (lambda c: p2wpkh(hash160_from_key(c))) if target == 'bech32' else addr_from_key
    for i in range(n):
        base = ((1 << 23) | secrets.randbelow(1 << 23)) << bits
        k = base + secrets.randbelow(1 << bits)
        addr = to_addr(k)
        pattern = addr[:5] if target == 'prefix' else addr
        found, tail, t0 = [], [], time.time()

        def on_line(s):
            tail.append(s); del tail[:-10]
            m = RX_KEY.search(s) or RX_CAND.search(s)
            if m:
                c = int(m.group(1), 16) % _N
                for cand in (c, (_N - c) % _N):
                    a = to_addr(cand)
                    if a == addr or (target == 'prefix' and a.startswith(pattern)): found.append(cand)
            return bool(found) or 'Range Finished' in s

        cmd = [exe, '-gpuId', '0', '-start', f'{base:x}', '-range', str(bits), '-stop'] + (['-random'] if random else [])
        listfile = None
        if target in ('bloom', 'tables'):
            # the planted address hidden among random ones, in the middle of the file
            listfile = os.path.join(os.path.dirname(exe), f'selftest_{os.getpid()}.txt')
            lines = [b58check(b'\x00' + secrets.token_bytes(20)) for _ in range(decoys)]
            lines.insert(len(lines) // 2, addr)
            with open(listfile, 'w') as fl: fl.write('\n'.join(lines) + '\n')
            cmd += ['-i', listfile] + (['-nobloom'] if target == 'tables' else [])
        else:
            cmd += [pattern]
        try:
            run_tool(cmd, timeout, on_line)
        finally:
            if listfile: os.remove(listfile)
        if target == 'prefix':
            ok = bool(found) and base <= found[0] < base + (1 << bits)
        else:
            ok = bool(found) and found[0] == k
        ok_all &= ok
        print(f'  [{i + 1}/{n}] {"PASS" if ok else "FAIL"}  key={k:x}  {time.time() - t0:.1f}s', flush=True)
        if not ok: print('    ' + '\n    '.join(tail))
    return ok_all


def speed(exe, start, bits, seconds, warmup):
    samples, t0 = [], time.time()

    def on_line(s):
        m = RX_SPEED.search(s)
        if m and time.time() - t0 > warmup: samples.append(float(m.group(1)))
        return False

    # a dummy address that will never be found
    run_tool([exe, '-gpuId', '0', '-start', f'{start:x}', '-range', str(bits),
              '1BitcoinEaterAddressDontSendf59kuE'], seconds, on_line)
    return statistics.median(samples) if samples else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['selftest', 'speed'])
    ap.add_argument('exes', nargs='+')
    ap.add_argument('--bits', type=int, default=36)
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--seconds', type=float, default=40)
    ap.add_argument('--warmup', type=float, default=15)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--random', action='store_true', help='selftest using the -random mode')
    ap.add_argument('--target', default='single', choices=['single', 'bech32', 'bloom', 'tables', 'prefix'],
                    help='selftest: which search path to test (see above)')
    ap.add_argument('--decoys', type=int, default=1000, help='selftest bloom/tables: random addresses next to the target')
    a = ap.parse_args()
    a.exes = [os.path.abspath(e) for e in a.exes]
    if a.mode == 'selftest':
        bad = [e for e in a.exes if not (print(e) or selftest(e, a.bits, a.n, timeout=300, random=a.random,
                                                             target=a.target, decoys=a.decoys))]
        sys.exit(1 if bad else 0)
    start = (1 << 70) + (secrets.randbelow(1 << 30) << 40)
    res = {e: [] for e in a.exes}
    for r in range(a.rounds):
        for e in a.exes:
            v = speed(e, start, 40, a.seconds, a.warmup)
            res[e].append(v)
            print(f'  round {r + 1}  {v:9.1f} MK/s  {e}', flush=True)
    base = statistics.median(res[a.exes[0]]) or 1
    print('\nmedian MK/s   vs first')
    for e, v in res.items():
        m = statistics.median(v)
        print(f'{m:10.1f}   {100 * (m / base - 1):+6.1f}%   {e}')


if __name__ == '__main__':
    main()

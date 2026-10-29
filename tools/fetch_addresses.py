#!/usr/bin/env python3
"""Download a Bitcoin address list from the Hugging Face dataset
pymmdrza/Bitcoin_Wallet_Addresses and write it as a plain .txt that
VanitySearch can read with -i.

The dataset is GATED: you must

  1. open https://huggingface.co/datasets/pymmdrza/Bitcoin_Wallet_Addresses
     while logged in and click "Agree and access repository",
  2. create a token at https://huggingface.co/settings/tokens (role: read),
  3. pass it with --token, or set the HF_TOKEN environment variable.

Examples
  python tools/fetch_addresses.py p2pkh                 # -> rich_p2pkh.txt  (1... addresses)
  python tools/fetch_addresses.py bech32 -o bc1.txt     # bc1q... addresses
  python tools/fetch_addresses.py p2pkh --token hf_xxx
  vanitysearch -start 1 -range 40 -stop -i rich_p2pkh.txt

Then search ONE address type per run: VanitySearch picks the type of the first
address and ignores the rest, so use the p2pkh file with a P2PKH search and the
bech32 file with a BECH32 search (the 'all' file mixes types).
"""
import argparse
import gzip
import os
import sys
import time
import urllib.request
import urllib.error

REPO = "pymmdrza/Bitcoin_Wallet_Addresses"
BASE = "https://huggingface.co/datasets/%s/resolve/main/" % REPO

# key -> (remote file, default output, one-line description)
FILES = {
    "p2pkh":  ("Latest_Rich_Bitcoin_P2PKH_Mmdrza.txt.gz",  "rich_p2pkh.txt",  "funded P2PKH (1...) addresses  [searchable]"),
    "bech32": ("Latest_Rich_Bitcoin_BECH32_Mmdrza.txt.gz", "rich_bech32.txt", "funded BECH32 (bc1q...) addresses  [searchable]"),
    "p2sh":   ("Latest_Rich_Bitcoin_P2SH_Mmdrza.txt.gz",   "rich_p2sh.txt",   "funded P2SH (3...) addresses  [NOT searchable by this tool]"),
    "all":    ("Latest_Rich_Bitcoin_Address_Mmdrza.txt.gz", "rich_all.txt",   "all funded addresses, types mixed"),
}


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def open_remote(url, token):
    """GET url with the HF token, following the redirect to the CDN by hand so the
    Authorization header is not leaked to the signed CDN URL."""
    headers = {"User-Agent": "vanitysearch-fetch", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = "Bearer " + token

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            return None  # stop, we follow manually without the auth header

    opener = urllib.request.build_opener(NoRedirect)
    try:
        return opener.open(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            loc = e.headers["Location"]
            return urllib.request.urlopen(
                urllib.request.Request(loc, headers={"User-Agent": "vanitysearch-fetch"}), timeout=60)
        if e.code in (401, 403):
            body = e.read(500).decode("utf-8", "replace")
            sys.exit("[ERROR] %d from Hugging Face: %s\n"
                     "        The dataset is gated. Accept its terms on the dataset page while logged in,\n"
                     "        then pass a read token with --token or the HF_TOKEN environment variable.\n"
                     "        Page: https://huggingface.co/datasets/%s" % (e.code, body.strip(), REPO))
        if e.code == 416:
            sys.exit("[ERROR] the server rejected the resume range; delete the partial .txt and retry")
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="p2pkh", choices=sorted(FILES),
                    help="which list to download (default: p2pkh)")
    ap.add_argument("-o", "--output", help="output .txt path (default depends on the list)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                    help="Hugging Face access token (or set HF_TOKEN)")
    ap.add_argument("--keep-gz", action="store_true", help="also keep the compressed .gz next to the .txt")
    ap.add_argument("--list", action="store_true", help="list the available files and exit")
    a = ap.parse_args()

    if a.list:
        for k, (fn, out, desc) in sorted(FILES.items()):
            print("  %-7s %s" % (k, desc))
        return

    remote, default_out, desc = FILES[a.which]
    out = os.path.abspath(a.output or default_out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    gz_path = out + ".gz"
    url = BASE + remote

    if a.which == "p2sh":
        print("[WARNING] P2SH (3...) addresses cannot be matched by this tool's kernel; downloading anyway.")
    if not a.token:
        print("[WARNING] no token given (--token / HF_TOKEN). The dataset is gated, so this will likely fail.")

    print("Downloading %s\n            %s -> %s" % (desc, remote, out))
    t0 = time.time()

    # 1) stream the .gz to disk (so a retry can resume, and decompression is separate)
    resp = open_remote(url, a.token)
    total = int(resp.headers.get("Content-Length") or 0)
    done = 0
    last = 0.0
    with open(gz_path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 0.5:
                last = now
                rate = done / (now - t0 + 1e-9)
                pct = (" %5.1f%%" % (100.0 * done / total)) if total else ""
                sys.stderr.write("\r  %s%s  %s/s   " % (human(done), pct, human(rate)))
                sys.stderr.flush()
    resp.close()
    sys.stderr.write("\r  downloaded %s in %.0f s%s\n" % (human(done), time.time() - t0, " " * 20))

    # 2) stream-decompress to the plain .txt and count the addresses
    print("Decompressing...")
    n = 0
    with gzip.open(gz_path, "rb") as src, open(out, "wb") as dst:
        while True:
            block = src.read(1 << 20)
            if not block:
                break
            dst.write(block)
            n += block.count(b"\n")
    if not a.keep_gz:
        os.remove(gz_path)

    size = os.path.getsize(out)
    print("Wrote %s (%s, ~%d addresses)" % (out, human(size), n))
    print("Search it with:  vanitysearch -start 1 -range 40 -stop -i %s   (one address type per run)" % out)


if __name__ == "__main__":
    main()

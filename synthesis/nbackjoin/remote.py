"""Read members out of remote zip archives with HTTP range requests.

Both sources serve multi-GB archives of which only the n-back files are needed.
``zipfile`` verifies each member's CRC-32 as it is read, so an extracted file is
byte-identical to the archived one. Pure standard library (no curl), so it behaves the
same on macOS, Linux and Windows.
"""
import http.client
import io
import shutil
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

TIMEOUT = 120          # s without any data before a connection counts as hung
ATTEMPTS = 8           # per request
CHUNK = 1 << 20


def _wait(attempt, err):
    """Back-off before retry `attempt` (0-based); honours Retry-After on HTTP 429/503."""
    if isinstance(err, urllib.error.HTTPError) and err.code in (429, 503):
        try:
            return min(600.0, float(err.headers.get("Retry-After", "")))
        except ValueError:
            pass
        return min(600.0, 30.0 * (attempt + 1))
    return min(300.0, 5.0 * 2 ** attempt)


def _retrying(fn, what):
    for attempt in range(ATTEMPTS):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404, 410, 416) or attempt == ATTEMPTS - 1:
                raise
            err = e
        except (OSError, http.client.HTTPException) as e:
            if attempt == ATTEMPTS - 1:
                raise
            err = e
        dt = _wait(attempt, err)
        print(f"  retry {attempt + 1}/{ATTEMPTS - 1} in {dt:.0f} s: {what}: {err!r}", flush=True)
        time.sleep(dt)


class HTTPRange(io.RawIOBase):
    """A seekable, read-only file object over an HTTP resource that accepts Range."""

    def __init__(self, url):
        self.pos = 0

        def head():
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=TIMEOUT) as r:
                return int(r.headers["Content-Length"]), r.url
        self.size, self.url = _retrying(head, f"HEAD {url}")

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})

        def get():
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                if r.status != 206:
                    raise IOError(f"server ignored the Range header (HTTP {r.status})")
                data = r.read()
            if len(data) != n:
                raise IOError(f"short read: {len(data)} of {n} bytes")
            return data
        data = _retrying(get, f"bytes {self.pos}-{self.pos + n - 1} of {self.url}")
        b[:n] = data
        self.pos += n
        return n


def download_range(url, start, end, part):
    """Append bytes start..end (inclusive) of ``url`` to ``part``, resuming after drops.

    Each retry restarts from the bytes already on disk, so a connection that hangs or
    breaks halfway loses nothing.
    """
    total = end - start + 1
    part = Path(part)
    stalled = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(part, "ab") as f:
                rng = r.headers.get("Content-Range", "")
                if r.status != 206 or not rng.startswith(f"bytes {start + have}-"):
                    raise IOError(f"unexpected response to Range request: HTTP {r.status} {rng!r}")
                while chunk := r.read(CHUNK):
                    f.write(chunk)
            stalled = 0
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404, 410, 416) or stalled >= ATTEMPTS:
                raise
            dt = _wait(stalled, e)
            stalled += 1
            print(f"  retry in {dt:.0f} s: {part.name}: {e!r}", flush=True)
            time.sleep(dt)
        except (OSError, http.client.HTTPException) as e:
            now = part.stat().st_size if part.exists() else 0
            stalled = 0 if now > have else stalled + 1   # progress resets the budget
            if stalled > ATTEMPTS:
                raise
            dt = _wait(min(stalled, 5), e)
            print(f"  resume in {dt:.0f} s at {now}/{total} bytes: {part.name}: {e!r}", flush=True)
            time.sleep(dt)


def fetch_member(url, z, info, out):
    """Download one member's compressed bytes (resumable), inflate, check CRC-32.

    More robust than streaming through ``zipfile`` for multi-GB pulls: Zenodo connections
    occasionally hang, and the download resumes from the partial file.
    """
    out = Path(out)
    if out.exists() and out.stat().st_size == info.file_size:
        return False
    z.fp.seek(info.header_offset)
    local = z.fp.read(30)
    if local[:4] != b"PK\x03\x04":
        raise ValueError(f"{info.filename}: bad local header")
    start = info.header_offset + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little")
    end = start + info.compress_size - 1
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".deflate")
    if part.exists() and part.stat().st_size > info.compress_size:
        part.unlink()                                  # corrupt leftover
    if info.compress_size:
        download_range(url, start, end, part)
    else:
        part.touch()
    if part.stat().st_size != info.compress_size:
        raise IOError(f"{info.filename}: incomplete after retries")
    crc, inflater = 0, zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    tmp = out.with_suffix(out.suffix + ".part")
    with open(part, "rb") as src, open(tmp, "wb") as dst:
        while chunk := src.read(8 << 20):
            data = inflater.decompress(chunk) if inflater else chunk
            crc = zlib.crc32(data, crc)
            dst.write(data)
        if inflater:
            data = inflater.flush()
            crc = zlib.crc32(data, crc)
            dst.write(data)
    if crc != info.CRC or tmp.stat().st_size != info.file_size:
        tmp.unlink()
        part.unlink()
        raise IOError(f"{info.filename}: CRC/size mismatch (partial download discarded; rerun to fetch again)")
    tmp.replace(out)
    part.unlink()
    return True


def open_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HTTPRange(url), buffer_size=8 << 20))


def extract(z, members, dest):
    """Extract ``members`` (ZipInfo) under ``dest``, skipping files already complete."""
    dest = Path(dest)
    for info in members:
        out = dest / info.filename
        if out.exists() and out.stat().st_size == info.file_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        with z.open(info) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
        tmp.replace(out)
        yield out

"""Read members out of remote zip archives with HTTP range requests.

Both sources serve multi-GB archives of which only the n-back files are needed.
``zipfile`` verifies each member's CRC-32 as it is read, so an extracted file is
byte-identical to the archived one.
"""
import http.client
import io
import shutil
import subprocess
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path


class HTTPRange(io.RawIOBase):
    """A seekable, read-only file object over an HTTP resource that accepts Range."""

    def __init__(self, url):
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url

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
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except (OSError, http.client.HTTPException):
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_member(url, z, info, out):
    """Download one deflated member's bytes with resumable curl, inflate, check CRC-32.

    More robust than streaming through ``zipfile`` for multi-GB pulls: Zenodo connections
    occasionally hang, and curl retries and resumes from the partial file.
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
    for attempt in range(20):
        have = part.stat().st_size if part.exists() else 0
        if have >= info.compress_size:
            break
        subprocess.run(["curl", "-sSL", "--retry", "10", "--retry-all-errors", "--speed-limit", "20000",
                        "--speed-time", "60", "-r", f"{start + have}-{end}", "-o", "-", url],
                       stdout=open(part, "ab"), check=False)
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
        raise IOError(f"{info.filename}: CRC/size mismatch")
    tmp.rename(out)
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
        tmp.rename(out)
        yield out

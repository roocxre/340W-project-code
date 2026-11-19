"""Read members out of remote zip archives with HTTP range requests.

Both sources serve multi-GB archives of which only the n-back files are needed.
``zipfile`` verifies each member's CRC-32 as it is read, so an extracted file is
byte-identical to the archived one. Pure standard library (no curl), so it behaves the
same on macOS, Linux and Windows.

Certificates. Every request tries two trust stores: the operating system's own
verifier (``truststore``: Windows CryptoAPI, macOS Keychain, Linux's CA bundle, so it
behaves like a browser and accepts corporate or antivirus roots installed there), then
Python's default store plus Mozilla's roots (``certifi``). If neither accepts a
server's certificate (expired, misconfigured, or an interception root nobody
installed), downloads whose content is pinned in ``data_sha256.txt`` continue without
TLS verification and every file is checked against its SHA-256 instead; anything else
stops with ``CertificateError``.
"""
import hashlib
import http.client
import io
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path

TIMEOUT = 120          # s without any data before a connection counts as hung
ATTEMPTS = 8           # per request
CHUNK = 1 << 20
REPO = Path(__file__).resolve().parents[2]
SUMS = Path(__file__).with_name("data_sha256.txt")   # repo-relative path -> SHA-256 of every data file


class CertificateError(Exception):
    """No trust store accepts a server's certificate. Not an OSError, so it is not retried."""


class IntegrityError(Exception):
    """A downloaded file does not match its pinned SHA-256."""


# ---------------------------------------------------------------------------- trust
_lock = threading.Lock()
_stores = None      # [(label, SSLContext)], built on first use
_host = {}          # host -> (label, context, verified) that last worked
_pinned = None


def trust_stores():
    global _stores
    with _lock:
        if _stores is None:
            _stores = []
            try:
                import truststore
                _stores.append(("system trust store", truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)))
            except ImportError:
                pass
            ctx = ssl.create_default_context()          # Python's default paths, incl. SSL_CERT_FILE
            try:
                import certifi
                ctx.load_verify_locations(certifi.where())
                _stores.append(("Mozilla roots (certifi)", ctx))
            except ImportError:
                _stores.append(("Python default store", ctx))
        return _stores


def pinned_sha256(path):
    """The pinned SHA-256 of a data file (by its path in the repo), or None."""
    global _pinned
    if _pinned is None:
        rows = (ln.split("  ", 1) for ln in SUMS.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#"))
        _pinned = {rel: sha for sha, rel in rows}
    try:
        return _pinned.get(Path(path).resolve().relative_to(REPO).as_posix())
    except ValueError:
        return None


def _cert_failure(e):
    r = getattr(e, "reason", e)
    if isinstance(r, ssl.SSLCertVerificationError):
        return getattr(r, "verify_message", None) or str(r)
    if isinstance(r, ssl.SSLError) and "CERTIFICATE" in str(r).upper():
        return str(r)
    return None


def urlopen(req, timeout=TIMEOUT, pinned=False):
    """urllib.request.urlopen through the first trust store that accepts the server.

    pinned: the caller checks everything it downloads against data_sha256.txt, so a
    certificate no store accepts is not fatal (see the module docstring)."""
    if isinstance(req, str):
        req = urllib.request.Request(req)
    host = urllib.parse.urlsplit(req.full_url).hostname
    known = _host.get(host)
    order = ([known[:2]] if known and known[2] else []) + [s for s in trust_stores() if not known or s[1] is not known[1]]
    failed = []
    for label, ctx in order:
        try:
            r = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        except (urllib.error.URLError, ssl.SSLError) as e:
            why = _cert_failure(e)
            if why is None:
                raise
            failed.append(f"{label}: {why}")
            continue
        _host[host] = (label, ctx, True)
        return r
    if not pinned:
        raise CertificateError(f"{host}: no trust store accepts its certificate ({'; '.join(failed)}). "
                               "On a network that intercepts HTTPS, set SSL_CERT_FILE to its CA bundle.")
    with _lock:
        if host not in _host or _host[host][2]:
            _host[host] = ("unverified", ssl._create_unverified_context(), False)
            print(f"WARNING  {host}: no trust store accepts its certificate ({'; '.join(failed)}). "
                  f"Continuing without TLS verification: every file from it is checked against its "
                  f"pinned SHA-256 ({SUMS.relative_to(REPO).as_posix()}) instead.", flush=True)
    return urllib.request.urlopen(req, timeout=timeout, context=_host[host][1])


def trusted_by(url):
    """Label of the trust store that last accepted url's host ("unverified" after a fallback)."""
    known = _host.get(urllib.parse.urlsplit(url).hostname)
    return known[0] if known else None


def verified(url):
    """True unless requests to url's host fell back to unverified TLS."""
    known = _host.get(urllib.parse.urlsplit(url).hostname)
    return known is None or known[2]


def check_sha256(path, sha, url):
    """Compare a finished download with its pin. Fatal if the connection was unverified; a
    warning otherwise (a verified server that changed its file is the source's choice)."""
    want = pinned_sha256(path)
    if want == sha:
        return
    if not verified(url):
        raise IntegrityError(f"{Path(path).name}: " + (f"SHA-256 {sha} does not match the pinned {want}"
                             if want else "no pinned SHA-256, so it cannot be checked without verified TLS"))
    if want:
        print(f"WARNING  {Path(path).name}: SHA-256 differs from the pinned one; the source changed "
              "this file since the committed results were made, so they may not reproduce exactly.", flush=True)


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

    def __init__(self, url, pinned=False):
        self.pos, self.pinned = 0, pinned

        def head():
            with urlopen(urllib.request.Request(url, method="HEAD"), pinned=pinned) as r:
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
            with urlopen(req, pinned=self.pinned) as r:
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


def download_range(url, start, end, part, pinned=False):
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
            with urlopen(req, pinned=pinned) as r, open(part, "ab") as f:
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
    """Download one member's compressed bytes (resumable), inflate, check CRC-32 and SHA-256.

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
        download_range(url, start, end, part, pinned=True)
    else:
        part.touch()
    if part.stat().st_size != info.compress_size:
        raise IOError(f"{info.filename}: incomplete after retries")
    crc, h = 0, hashlib.sha256()
    inflater = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    tmp = out.with_suffix(out.suffix + ".part")
    with open(part, "rb") as src, open(tmp, "wb") as dst:
        while chunk := src.read(8 << 20):
            data = inflater.decompress(chunk) if inflater else chunk
            crc = zlib.crc32(data, crc)
            h.update(data)
            dst.write(data)
        if inflater:
            data = inflater.flush()
            crc = zlib.crc32(data, crc)
            h.update(data)
            dst.write(data)
    try:
        if crc != info.CRC or tmp.stat().st_size != info.file_size:
            raise IOError(f"{info.filename}: CRC/size mismatch (partial download discarded; rerun to fetch again)")
        check_sha256(out, h.hexdigest(), url)
    except (IOError, IntegrityError):
        tmp.unlink()
        part.unlink()
        raise
    tmp.replace(out)
    part.unlink()
    return True


def open_zip(url, pinned=False):
    """A remote zip. pinned=True only if every member read from it is checked by check_sha256."""
    return zipfile.ZipFile(io.BufferedReader(HTTPRange(url, pinned), buffer_size=8 << 20))


def extract(z, members, dest, url):
    """Extract ``members`` (ZipInfo) of ``z`` (opened from ``url``) under ``dest``, skipping
    files already complete; each new file is checked against its pinned SHA-256."""
    dest = Path(dest)
    for info in members:
        out = dest / info.filename
        if out.exists() and out.stat().st_size == info.file_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        h = hashlib.sha256()
        with z.open(info) as src, open(tmp, "wb") as dst:
            while chunk := src.read(8 << 20):
                h.update(chunk)
                dst.write(chunk)
        try:
            check_sha256(out, h.hexdigest(), url)
        except IntegrityError:
            tmp.unlink()
            raise
        tmp.replace(out)
        yield out


def download(url, out):
    """Download a whole (small) file to ``out``, checked against its pinned SHA-256."""
    out = Path(out)
    if out.exists() and pinned_sha256(out) and _sha256(out) == pinned_sha256(out):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")

    def get():
        h = hashlib.sha256()
        with urlopen(url, pinned=True) as r, open(tmp, "wb") as f:
            while chunk := r.read(CHUNK):
                h.update(chunk)
                f.write(chunk)
        return h.hexdigest()
    sha = _retrying(get, f"GET {url}")
    try:
        check_sha256(out, sha, url)
    except IntegrityError:
        tmp.unlink()
        raise
    tmp.replace(out)
    return out


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

"""Certificate handling of nbackjoin.remote against a local HTTPS server.

The server's certificate is signed by a test CA (tls/ca.pem) that no real trust store
holds, which is the situation of a data server whose root a machine lacks."""
import hashlib
import http.server
import io
import ssl
import sys
import threading
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nbackjoin import remote

TLS = Path(__file__).with_name("tls")
NOTEBOOK = b"notebook bytes " * 1000
SET = bytes(range(256)) * 400


def _zip():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("sub-01/ses-S1/eeg/zeroBACK.set", SET, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("readme.txt", b"stored member", compress_type=zipfile.ZIP_STORED)
    return b.getvalue()


FILES = {"/notebook.mat": NOTEBOOK, "/sub-01.zip": _zip()}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body_wanted):
        data = FILES.get(self.path)
        if data is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), int(b) if b else len(data) - 1
            part = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            part = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        if body_wanted:
            self.wfile.write(part)

    def do_HEAD(self):
        self._send(False)

    def do_GET(self):
        self._send(True)


@pytest.fixture
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(TLS / "server.pem")
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    httpd.handle_error = lambda *a: None
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"https://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    """Real trust stores, no remembered hosts, and pins relative to tmp_path."""
    monkeypatch.setattr(remote, "_stores", None)
    monkeypatch.setattr(remote, "_host", {})
    monkeypatch.setattr(remote, "REPO", tmp_path)
    monkeypatch.setattr(remote, "SUMS", tmp_path / "sums.txt")
    monkeypatch.setattr(remote, "_pinned", {})


def pin(rel, data):
    remote._pinned[rel] = hashlib.sha256(data).hexdigest()


def trust_test_ca(monkeypatch):
    nothing = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)           # trusts no root at all
    monkeypatch.setattr(remote, "_stores", [("empty store", nothing),
                                            ("test CA", ssl.create_default_context(cafile=TLS / "ca.pem"))])


def test_untrusted_unpinned_request_fails_fast(server):
    with pytest.raises(remote.CertificateError, match="no trust store accepts"):
        remote.HTTPRange(server + "/sub-01.zip")                 # not retried: CertificateError is no OSError


def test_untrusted_pinned_download_is_checked_by_sha256(server, tmp_path, capsys):
    pin("notebook.mat", NOTEBOOK)
    out = remote.download(server + "/notebook.mat", tmp_path / "notebook.mat")
    assert out.read_bytes() == NOTEBOOK
    assert remote.trusted_by(server) == "unverified" and not remote.verified(server)
    assert "Continuing without TLS verification" in capsys.readouterr().out


@pytest.mark.parametrize("pinned", ["wrong", "missing"])
def test_untrusted_download_without_matching_pin_is_refused(server, tmp_path, pinned):
    if pinned == "wrong":
        pin("notebook.mat", b"other bytes")
    with pytest.raises(remote.IntegrityError):
        remote.download(server + "/notebook.mat", tmp_path / "notebook.mat")
    assert list(tmp_path.iterdir()) == []                        # nothing left behind


def test_untrusted_zip_members_are_checked_by_sha256(server, tmp_path):
    url = server + "/sub-01.zip"
    pin("x/zeroBACK.set", SET)
    pin("y/readme.txt", b"stored member")
    z = remote.open_zip(url, pinned=True)
    set_info, txt_info = z.infolist()
    assert remote.fetch_member(url, z, set_info, tmp_path / "x" / "zeroBACK.set")
    assert (tmp_path / "x" / "zeroBACK.set").read_bytes() == SET
    assert [p.name for p in remote.extract(z, [txt_info], tmp_path / "y", url)] == ["readme.txt"]
    remote._pinned["x/zeroBACK.set"] = "0" * 64                  # a tampered member is discarded
    (tmp_path / "x" / "zeroBACK.set").unlink()
    with pytest.raises(remote.IntegrityError):
        remote.fetch_member(url, z, set_info, tmp_path / "x" / "zeroBACK.set")
    assert not (tmp_path / "x" / "zeroBACK.set").exists()


def test_second_trust_store_is_used_when_the_first_rejects(server, tmp_path, monkeypatch, capsys):
    trust_test_ca(monkeypatch)
    remote.HTTPRange(server + "/sub-01.zip")
    assert remote.trusted_by(server) == "test CA" and remote.verified(server)
    # verified source whose file changed: kept, with a warning (not fatal)
    pin("notebook.mat", b"older version")
    assert remote.download(server + "/notebook.mat", tmp_path / "notebook.mat").read_bytes() == NOTEBOOK
    out = capsys.readouterr().out
    assert "differs from the pinned" in out and "without TLS verification" not in out


def test_pinned_manifest_covers_every_fetched_file():
    sums = Path(remote.__file__).with_name("data_sha256.txt").read_text().splitlines()
    rows = [ln.split("  ", 1) for ln in sums if not ln.startswith("#")]
    assert all(len(sha) == 64 for sha, _ in rows)
    paths = [p for _, p in rows]
    assert len(paths) == len(set(paths)) == 78 + 29 * 18 + 1
    assert sum(p.startswith("eeg-analysis/Cleaned Data/VP0") and p.endswith("_nback.mat") for p in paths) == 78
    assert sum(p.startswith("synthesis/data/cogbci/sub-") for p in paths) == 29 * 18
    assert "synthesis/data/cogbci/notebook.mat" in paths

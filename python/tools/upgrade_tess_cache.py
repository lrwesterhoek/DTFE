#!/usr/bin/env python3
"""Convert a tessellation cache ('--tessellation-cache' folder) to the CHUNKED body.

Since 2026-10-02 the binaries write a cached tessellation's body as independent gzip members of
16 MB (tessellation_cache.h, ChunkDeflateBuf), which a load inflates on several threads -- about 3 s
of the 8.4 s a 6M-vertex TNG100-3 partition took. Older files hold one gzip stream; they still load,
on one core. This tool rewrites them: the header (magic, length, descriptor) is copied as it is, the
body is inflated and recompressed chunk by chunk, the result is checked (the inflated content's
SHA-256 must be the old one's) and only then renamed over the old file. Files already chunked are
left alone; occupancy maps ('.occ') and globals records do not change.

    python3 python/tools/upgrade_tess_cache.py "/Volumes/Samsung T7/Illustris TNG/.dtfe-tessellation-cache"

Run it only when no OLD binary (built before the chunked format) uses that folder -- a query server
started before the update cannot read converted files. -j sets the number of files converted at once.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

TESS_MAGIC = b"DTFETESS"
CHUNK_MARKER = b"DTFECHK1"
CHUNK_FOOTER = b"DTFECHKE"
CHUNK_BYTES = 16 << 20
LEVEL = 1              # the binaries' DEFLATE_LEVEL


def _header(f) -> bytes | None:
    magic = f.read(8)
    if magic != TESS_MAGIC:
        return None
    raw = f.read(4)
    if len(raw) != 4:
        return None
    (n,) = struct.unpack("<I", raw)
    if not 0 < n < (1 << 20):
        return None
    desc = f.read(n)
    return magic + raw + desc if len(desc) == n else None


def is_chunked(path: Path) -> bool | None:
    """True / False for a chunked / single-stream file, None when it is not a tessellation file."""
    with open(path, "rb") as f:
        if _header(f) is None:
            return None
        return f.read(8) == CHUNK_MARKER


def convert(path: Path) -> str:
    """Rewrite one single-stream file as chunks; returns what happened."""
    path = Path(path)
    tmp = path.with_name(path.name + ".chunked.part")
    with open(path, "rb") as f:
        head = _header(f)
        if head is None:
            return f"skipped (not a tessellation file): {path.name}"
        body_start = f.tell()
        if f.read(8) == CHUNK_MARKER:
            return f"already chunked: {path.name}"
        f.seek(body_start)
        sha_old = hashlib.sha256()
        sizes: list[tuple[int, int]] = []
        try:
            with open(tmp, "wb") as out:
                out.write(head)
                out.write(CHUNK_MARKER)
                written = 0
                inflater = zlib.decompressobj(15 + 16)
                pending = bytearray()

                def emit(chunk: bytes):
                    nonlocal written
                    c = zlib.compressobj(LEVEL, zlib.DEFLATED, 15 + 16)
                    member = c.compress(chunk) + c.flush()
                    out.write(member)
                    sizes.append((len(member), len(chunk)))
                    written += len(member)

                while True:
                    block = f.read(8 << 20)
                    if not block:
                        break
                    data = inflater.decompress(block)
                    while inflater.unconsumed_tail:
                        data += inflater.decompress(inflater.unconsumed_tail)
                    sha_old.update(data)
                    pending += data
                    while len(pending) >= CHUNK_BYTES:
                        emit(bytes(pending[:CHUNK_BYTES]))
                        del pending[:CHUNK_BYTES]
                    if inflater.eof:
                        break
                if not inflater.eof:
                    raise ValueError("the old body ends before its gzip stream does (a truncated file)")
                if pending:
                    emit(bytes(pending))
                index_offset = len(head) + len(CHUNK_MARKER) + written
                out.write(struct.pack("<Q", len(sizes)))
                for csize, usize in sizes:
                    out.write(struct.pack("<QQ", csize, usize))
                out.write(struct.pack("<Q", index_offset))
                out.write(CHUNK_FOOTER)
                out.flush()
                os.fsync(out.fileno())
            # check: the new file's chunks inflate to exactly the old content
            sha_new = hashlib.sha256()
            with open(tmp, "rb") as g:
                g.seek(len(head) + len(CHUNK_MARKER))
                for csize, usize in sizes:
                    d = zlib.decompress(g.read(csize), 15 + 16)
                    if len(d) != usize:
                        raise ValueError("a chunk inflates to the wrong size")
                    sha_new.update(d)
            if sha_new.digest() != sha_old.digest():
                raise ValueError("the chunked content differs from the old one")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            return f"FAILED {path.name}: {e}"
    st = path.stat()
    os.replace(tmp, path)
    os.utime(path, (st.st_atime, st.st_mtime))
    return f"converted: {path.name} ({len(sizes)} chunks)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cache", help="a '--tessellation-cache' folder (or one .tess file)")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="files converted at once (default: half the cores)")
    ap.add_argument("--dry-run", action="store_true", help="only count the files to convert")
    a = ap.parse_args(argv)
    root = Path(a.cache).expanduser()
    files = [root] if root.is_file() else sorted(root.glob("tess_*.tess"))
    todo = [p for p in files if is_chunked(p) is False]
    print(f"{len(files)} tessellation files, {len(todo)} in the single-stream format")
    if a.dry_run or not todo:
        return 0
    failed = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for fut in as_completed([pool.submit(convert, p) for p in todo]):
            msg = fut.result()
            failed += msg.startswith("FAILED")
            print(msg, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Lossless streaming Zstandard archives and byte-exact model XOR deltas.

Every delta references the immutable legacy checkpoint, never another delta.
Optimizer/scheduler/RNG bytes are compressed unchanged, not quantized/restarted.
"""

import io
import json
import tarfile
from pathlib import Path

import numpy as np

from .checkpoints import sha256


class CodecError(ValueError):
    pass


class XorReader:
    def __init__(self, source, base):
        self.source, self.base = source, base

    def read(self, size):
        data = self.source.read(size)
        reference = self.base.read(len(data)).ljust(len(data), b"\0")
        return np.bitwise_xor(
            np.frombuffer(data, dtype=np.uint8), np.frombuffer(reference, dtype=np.uint8)
        ).tobytes()


def pack(directory, destination, base_dir=None):
    import zstandard as zstd

    directory, destination = Path(directory), Path(destination)
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    records = []
    for i, path in enumerate(files):
        if path.is_symlink():
            raise CodecError("Checkpoint symlinks are forbidden")
        name = str(path.relative_to(directory))
        base = Path(base_dir) / name if base_dir else None
        delta = name.endswith(".safetensors") and base is not None and base.is_file()
        records.append(
            {
                "path": name,
                "member": f"payload/{i:05d}",
                "size": path.stat().st_size,
                "encoding": "xor" if delta else "plain",
                "base_sha256": sha256(base) if delta else None,
            }
        )
    header = json.dumps({"version": 1, "files": records}).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("wb") as raw,
        zstd.ZstdCompressor(level=3, threads=2).stream_writer(raw) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            archive.copybufsize = 4 * 1024 * 1024
            info = tarfile.TarInfo("codec.json")
            info.size = len(header)
            archive.addfile(info, io.BytesIO(header))
            for record, path in zip(records, files):
                info = tarfile.TarInfo(record["member"])
                info.size = record["size"]
                with path.open("rb") as stream:
                    if record["encoding"] == "xor":
                        with (Path(base_dir) / record["path"]).open("rb") as reference:
                            archive.addfile(info, XorReader(stream, reference))
                    else:
                        archive.addfile(info, stream)
    return destination


def unpack(archive_path, destination, max_bytes, base_dir=None):
    import zstandard as zstd

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with Path(archive_path).open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            members = iter(archive)
            header = next(members, None)
            if header is None or header.name != "codec.json" or header.size > 4_000_000:
                raise CodecError("Missing/bounded codec header")
            data = json.loads(archive.extractfile(header).read())
            records = data.get("files", [])
            if data.get("version") != 1 or sum(r["size"] for r in records) > max_bytes:
                raise CodecError("Invalid codec version or expanded size")
            expected = {r["member"]: r for r in records}
            if len(expected) != len(records):
                raise CodecError("Duplicate archive members")
            paths = set()
            for member in members:
                record = expected.pop(member.name, None)
                if record is None or not member.isfile() or member.size != record["size"]:
                    raise CodecError("Unexpected archive member")
                target = destination / record["path"]
                if record["path"] in paths or not target.resolve().is_relative_to(
                    destination.resolve()
                ):
                    raise CodecError("Duplicate/unsafe destination path")
                paths.add(record["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                base = None
                if record["encoding"] == "xor":
                    if base_dir is None:
                        raise CodecError("Immutable delta base is required")
                    base_path = Path(base_dir) / record["path"]
                    if not base_path.resolve().is_relative_to(Path(base_dir).resolve()):
                        raise CodecError("Unsafe base path")
                    if sha256(base_path) != record["base_sha256"]:
                        raise CodecError("Immutable delta base checksum mismatch")
                    base = base_path.open("rb")
                elif record["encoding"] != "plain":
                    raise CodecError("Unknown encoding")
                try:
                    with archive.extractfile(member) as source, target.open("xb") as output:
                        while block := source.read(4 * 1024 * 1024):
                            if base:
                                reference = base.read(len(block)).ljust(len(block), b"\0")
                                block = np.bitwise_xor(
                                    np.frombuffer(block, dtype=np.uint8),
                                    np.frombuffer(reference, dtype=np.uint8),
                                ).tobytes()
                            output.write(block)
                finally:
                    if base:
                        base.close()
            if expected:
                raise CodecError("Archive is incomplete")

import io
import json
import tarfile

import pytest
from _storage_helpers import MemoryBackend

from arabic_summarizer.checkpoint_codec import CodecError, pack, unpack
from arabic_summarizer.checkpoints import seal, valid
from arabic_summarizer.io import atomic_json
from arabic_summarizer.storage import ManagedStore, Quota, StorageBlocked, StoragePolicy


def checkpoint(path, fingerprint="run", step=10):
    path.mkdir(parents=True)
    for name in ["config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
        (path / name).write_bytes(b"real fixture bytes " * 100)
    (path / "model.safetensors").write_bytes(bytes(range(256)) * 1000)
    atomic_json(path / "trainer_state.json", {"global_step": step})
    seal(path, fingerprint)
    return path


def test_reported_quota_blocks_before_upload(tmp_path):
    backend = MemoryBackend(external_usage=12_740_000_000)
    policy = StoragePolicy()
    q = backend.quota()
    with pytest.raises(StorageBlocked, match="Account quota blocks"):
        policy.require(q, 4_410_000_000)
    plan = policy.plan(
        q, 4_410_000_000, 1_470_000_000, 4_740_000_000, protected_bytes=4_410_000_000
    )
    assert plan["account_minus_project_bytes"] == 8_000_000_000
    assert plan["next_full_write_shortfall_bytes"] > 3_000_000_000
    assert backend.uploads == 0 and backend.deleted == []


def test_actual_quota_and_trash_count_not_virtual_disk_or_40gb_limit():
    q = Quota.from_api(
        {
            "storageQuota": {
                "limit": "16106127360",
                "usage": "12740000000",
                "usageInDrive": "12740000000",
                "usageInDriveTrash": "8000000000",
            }
        }
    )
    assert q.report()["drive_trash_bytes"] == 8_000_000_000
    assert q.report()["outside_drive_services_bytes"] == 0
    assert q.limit == 15 * 2**30
    with pytest.raises(StorageBlocked):
        StoragePolicy().require(q, 4_410_000_000)


@pytest.mark.parametrize(
    "data",
    [{"usage": "1"}, {"limit": "0", "usage": "0", "usageInDrive": "0", "usageInDriveTrash": "0"}],
)
def test_unknown_or_invalid_quota_fails_closed(data):
    with pytest.raises(StorageBlocked):
        Quota.from_api(data)


def test_xor_zstd_roundtrip_is_byte_exact_including_optimizer_rng(tmp_path):
    base = checkpoint(tmp_path / "base")
    current = checkpoint(tmp_path / "current", step=11)
    original = (current / "model.safetensors").read_bytes()
    (current / "model.safetensors").write_bytes(original[:-4] + b"last")
    seal(current, "run")
    packed = pack(current, tmp_path / "state.tar.zst", base)
    unpack(packed, tmp_path / "restored", sum(p.stat().st_size for p in current.iterdir()), base)
    for path in current.iterdir():
        assert path.read_bytes() == (tmp_path / "restored" / path.name).read_bytes()
    assert valid(tmp_path / "restored", "run")
    assert packed.stat().st_size < (current / "model.safetensors").stat().st_size


def test_wrong_base_is_rejected(tmp_path):
    base = checkpoint(tmp_path / "base")
    current = checkpoint(tmp_path / "current")
    packed = pack(current, tmp_path / "state.tar.zst", base)
    (base / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(CodecError, match="base checksum"):
        unpack(packed, tmp_path / "restore", 1_000_000, base)


def test_unsafe_codec_member_is_rejected(tmp_path):
    import zstandard as zstd

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        header = json.dumps(
            {
                "version": 1,
                "files": [
                    {"path": "../escape", "member": "payload/0", "size": 1, "encoding": "plain"}
                ],
            }
        ).encode()
        for name, data in [("codec.json", header), ("payload/0", b"x")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive = tmp_path / "bad.zst"
    archive.write_bytes(zstd.ZstdCompressor().compress(raw.getvalue()))
    with pytest.raises(CodecError, match="unsafe"):
        unpack(archive, tmp_path / "restore", 100)
    assert not (tmp_path / "escape").exists()


def test_replace_requires_validation_and_explicit_confirmation(tmp_path):
    backend = MemoryBackend()
    store = ManagedStore(
        backend, tmp_path / "local", "run", StoragePolicy(safety_margin_bytes=1000)
    )
    store.cleanup()  # Empty new store has nothing to retire.
    first = store.commit_checkpoint(checkpoint(tmp_path / "first"), "candidate", 10, {})
    second = store.commit_checkpoint(checkpoint(tmp_path / "second", step=11), "candidate", 11, {})
    assert backend.deleted == [] and first["id"] in backend.files
    with pytest.raises(StorageBlocked, match="confirmation"):
        store.cleanup()
    plan = store.cleanup_plan()
    store.cleanup(confirmation=plan["confirmation"])
    assert backend.deleted == [first["id"]]
    assert second["id"] in backend.files
    restored = store.restore(second, tmp_path / "restored")
    assert valid(restored, "run")
    assert store.restore(second, tmp_path / "restored") == restored


def test_corrupt_upload_never_retires_old_state(tmp_path):
    backend = MemoryBackend()
    store = ManagedStore(
        backend, tmp_path / "local", "run", StoragePolicy(safety_margin_bytes=1000)
    )
    old = store.commit_checkpoint(checkpoint(tmp_path / "old"), "a", 10, {})
    backend.corrupt_upload = True
    with pytest.raises(StorageBlocked, match="differs"):
        store.commit_checkpoint(checkpoint(tmp_path / "new"), "a", 11, {})
    assert backend.read_index()["latest"] == old and old["id"] in backend.files
    assert backend.deleted == []


def test_failed_index_publish_keeps_old_checkpoint(tmp_path):
    backend = MemoryBackend()
    store = ManagedStore(
        backend, tmp_path / "local", "run", StoragePolicy(safety_margin_bytes=1000)
    )
    old = store.commit_checkpoint(checkpoint(tmp_path / "old"), "a", 10, {})
    backend.fail_index = True
    with pytest.raises(OSError):
        store.commit_checkpoint(checkpoint(tmp_path / "new"), "a", 11, {})
    assert backend.read_index()["latest"] == old and backend.deleted == []


def test_changed_cleanup_plan_token_is_not_reused(tmp_path):
    backend = MemoryBackend()
    store = ManagedStore(
        backend, tmp_path / "local", "run", StoragePolicy(safety_margin_bytes=1000)
    )
    for i in (10, 11):
        store.commit_checkpoint(checkpoint(tmp_path / f"c{i}", step=i), "a", i, {})
    token = store.cleanup_plan()["confirmation"]
    store.commit_checkpoint(checkpoint(tmp_path / "c12", step=12), "a", 12, {})
    with pytest.raises(StorageBlocked, match="confirmation"):
        store.cleanup(confirmation=token)
    assert backend.deleted == []


def test_stale_quota_is_not_accepted():
    quota = Quota(15_000_000_000, 0, 0, 0, 0)
    with pytest.raises(StorageBlocked, match="stale"):
        StoragePolicy().require(quota, 1)


def test_no_upload_is_started_when_live_quota_drops(tmp_path):
    backend = MemoryBackend()
    store = ManagedStore(
        backend, tmp_path / "local", "run", StoragePolicy(safety_margin_bytes=1000)
    )
    old = store.commit_checkpoint(checkpoint(tmp_path / "old"), "a", 10, {})
    backend.external_usage = backend.limit
    with pytest.raises(StorageBlocked, match="quota blocks"):
        store.commit_checkpoint(checkpoint(tmp_path / "new"), "a", 11, {})
    assert backend.uploads == 1
    assert backend.read_index()["latest"] == old
    assert backend.deleted == []


def test_step10_compression_is_not_a_guarantee_of_1200_step_capacity():
    backend = MemoryBackend(limit=15 * 2**30, external_usage=4_910_000_000)
    plan = StoragePolicy().plan(
        backend.quota(),
        4_410_000_000,
        1_470_000_000,
        4_740_000_000,
        protected_bytes=4_410_000_000,
        current_archive_bytes=3_200_000_000,
        best_archive_bytes=900_000_000,
    )
    assert plan["measured_current_replacement_plan_fits"]
    assert not plan["full_plan_fits"]
    assert plan["uncompressed_lifecycle_account_peak_bytes"] == 16_641_000_000
    assert plan["minimum_additional_capacity_or_offload_bytes"] == 534_872_640

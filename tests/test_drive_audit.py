import copy

import pytest
from test_storage import checkpoint

from arabic_summarizer.drive_storage import APP, DriveBackend, account_audit
from arabic_summarizer.storage import StorageBlocked
from arabic_summarizer.trash_cleanup import apply_smoke_cleanup, preview_smoke_cleanup


class Request:
    def __init__(self, callback):
        self.callback = callback

    def execute(self):
        return self.callback()


class Service:
    def __init__(self):
        self.deleted = []
        self.rows = {
            "project": {"id": "project", "name": "full-v2-fp32-final-01", "parents": []},
            "experiments": {"id": "experiments", "name": "experiments", "parents": ["project"]},
            "smoke": {"id": "smoke", "name": "smoke", "parents": ["experiments"]},
            "c1": {"id": "c1", "name": "checkpoint-1", "parents": ["smoke"], "trashed": True},
            "old": {
                "id": "old",
                "name": "optimizer.pt",
                "parents": ["c1"],
                "trashed": True,
                "size": "100",
                "quotaBytesUsed": "8000000000",
                "md5Checksum": "old-md5",
            },
            "candidate": {"id": "candidate", "name": "arat5-5e5", "parents": ["experiments"]},
            "c10": {"id": "c10", "name": "checkpoint-10", "parents": ["candidate"]},
            "current": {
                "id": "current",
                "name": "optimizer.pt",
                "parents": ["c10"],
                "trashed": False,
                "size": "100",
                "quotaBytesUsed": "4740000000",
                "md5Checksum": "current-md5",
            },
            "unrelated": {
                "id": "unrelated",
                "name": "optimizer.pt",
                "parents": [],
                "trashed": True,
                "size": "10",
                "quotaBytesUsed": "10",
                "md5Checksum": "other-md5",
            },
        }

    def files(self):
        return self

    def about(self):
        return self

    def get(self, fileId=None, **kwargs):
        if fileId is None:
            return Request(
                lambda: {
                    "storageQuota": {
                        "limit": "15000000000",
                        "usage": "12740000010",
                        "usageInDrive": "12740000010",
                        "usageInDriveTrash": "8000000010",
                    }
                }
            )
        return Request(lambda: copy.deepcopy(self.rows[fileId]))

    def list(self, **kwargs):
        return Request(lambda: {"files": list(copy.deepcopy(self.rows).values())})

    def delete(self, fileId):
        return Request(lambda: self.deleted.append(fileId))


def test_trash_explains_gap_without_attributing_it_to_preexisting_files():
    service = Service()
    report = account_audit(service, "project")
    assert service.deleted == []
    assert report["totals_bytes"]["project_trash"] == 8_000_000_000
    assert report["totals_bytes"]["project_live"] == 4_740_000_000
    assert report["totals_bytes"]["other_trash"] == 10
    assert report["quota"]["outside_drive_services_bytes"] == 0


def test_only_obsolete_smoke_is_offered_and_confirmation_is_mandatory(tmp_path):
    service = Service()
    protected = checkpoint(tmp_path / "checkpoint-10")
    plan = preview_smoke_cleanup(account_audit(service, "project"), protected, "run")
    assert [r["id"] for r in plan["candidates"]] == ["old"]
    with pytest.raises(StorageBlocked, match="confirmation"):
        apply_smoke_cleanup(service, plan, "yes")
    assert service.deleted == []
    original = (protected / "optimizer.pt").read_bytes()
    result = apply_smoke_cleanup(service, plan, plan["confirmation"])
    assert result["deleted_ids"] == ["old"]
    assert service.deleted == ["old"]
    assert (protected / "optimizer.pt").read_bytes() == original


def test_protected_corruption_prevents_all_cleanup(tmp_path):
    service = Service()
    protected = checkpoint(tmp_path / "checkpoint-10")
    plan = preview_smoke_cleanup(account_audit(service, "project"), protected, "run")
    (protected / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(StorageBlocked, match="Protected checkpoint"):
        apply_smoke_cleanup(service, plan, plan["confirmation"])
    assert service.deleted == []


def test_candidate_metadata_or_ancestry_change_prevents_cleanup(tmp_path):
    service = Service()
    protected = checkpoint(tmp_path / "checkpoint-10")
    plan = preview_smoke_cleanup(account_audit(service, "project"), protected, "run")
    service.rows["c1"]["parents"] = []
    with pytest.raises(StorageBlocked, match="candidate changed"):
        apply_smoke_cleanup(service, plan, plan["confirmation"])
    assert service.deleted == []


def test_managed_delete_requires_confirmation_and_namespace_membership():
    service = Service()
    backend = DriveBackend(service, "run", folder_id="managed-folder")
    artifact = {"id": "current", "size": 100, "md5": "current-md5"}
    with pytest.raises(StorageBlocked, match="confirmation"):
        backend.delete_managed(artifact)
    with pytest.raises(StorageBlocked, match="belong"):
        backend.delete_managed(artifact, confirmation="DELETE current current-md5")
    assert service.deleted == []
    service.rows["current"]["parents"] = ["managed-folder"]
    service.rows["current"]["appProperties"] = {"app": APP, "run": "run", "kind": "full"}
    backend.delete_managed(artifact, confirmation="DELETE current current-md5")
    assert service.deleted == ["current"]

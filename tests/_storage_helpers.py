import copy
import hashlib
import time
from pathlib import Path

from arabic_summarizer.storage import Quota, StorageBlocked


class MemoryBackend:
    def __init__(self, limit=15_000_000_000, external_usage=0):
        self.limit, self.external_usage = limit, external_usage
        self.files, self.index, self.deleted = {}, None, []
        self.deleted_records = {}
        self.corrupt_upload = False
        self.fail_index = False
        self.uploads = 0

    def quota(self):
        usage = self.external_usage + sum(len(item["data"]) for item in self.files.values())
        return Quota(self.limit, usage, usage, self.external_usage, time.time())

    def managed_bytes(self):
        return sum(len(item["data"]) for item in self.files.values())

    def read_index(self):
        return copy.deepcopy(self.index)

    def write_index(self, value):
        if self.fail_index:
            raise OSError("interrupted index publication")
        self.index = copy.deepcopy(value)

    def upload(self, path, kind, experiment, step):
        self.uploads += 1
        data = Path(path).read_bytes()
        if self.corrupt_upload:
            data += b"corrupted"
        key = str(self.uploads)
        self.files[key] = {"data": data, "kind": kind, "experiment": experiment, "step": step}
        return self.stat(key)

    def stat(self, key):
        data = self.files[key]["data"]
        return {"id": key, "size": len(data), "md5": hashlib.md5(data).hexdigest()}

    def download(self, key, path):
        Path(path).write_bytes(self.files[key]["data"])

    def delete_managed(self, artifact, confirmation=None):
        if confirmation != "DELETE " + artifact["id"] + " " + artifact["md5"]:
            raise StorageBlocked("confirmation missing")
        if artifact["id"] not in self.files:
            return
        assert self.stat(artifact["id"]) == {k: artifact[k] for k in ("id", "size", "md5")}
        self.deleted_records[artifact["id"]] = self.files[artifact["id"]]
        self.deleted.append(artifact["id"])
        del self.files[artifact["id"]]

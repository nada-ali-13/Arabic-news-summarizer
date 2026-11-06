"""Google Drive API adapter. Credentials stay in the normal SDK, never in files/logs."""

import io
import json
import uuid

from .storage import Quota, StorageBlocked

FIELDS = "id,name,size,quotaBytesUsed,md5Checksum,parents,appProperties,trashed"
APP = "arabic-news-summarizer-v1"


def quote(value):
    return value.replace("\\", "\\\\").replace("'", "\\'")


def colab_service(allow_legacy_delete=False):
    import google.auth
    import google_auth_httplib2
    import httplib2
    from google.colab import auth
    from googleapiclient.discovery import build

    auth.authenticate_user()
    credentials, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/drive.metadata.readonly",
            "https://www.googleapis.com/auth/drive"
            if allow_legacy_delete
            else "https://www.googleapis.com/auth/drive.file",
        ]
    )
    http = google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=60))
    return build("drive", "v3", http=http, cache_discovery=False)


def account_audit(service, project_folder_id=None, max_pages=1000):
    """Read-only metadata audit, including owned trashed files and ancestor IDs.

    Never classifies files as disposable from their names alone. Output should
    remain private; no media content or credentials are fetched.
    """
    quota = Quota.from_api(service.about().get(fields="storageQuota").execute())
    files, token = {}, None
    for _ in range(max_pages):
        page = (
            service.files()
            .list(
                q="'me' in owners",
                spaces="drive",
                pageSize=1000,
                pageToken=token,
                fields="incompleteSearch,nextPageToken,files(" + FIELDS + ",mimeType,ownedByMe)",
            )
            .execute()
        )
        if page.get("incompleteSearch"):
            raise StorageBlocked("Drive metadata search was incomplete; no cleanup plan is safe")
        files.update({item["id"]: item for item in page.get("files", [])})
        token = page.get("nextPageToken")
        if not token:
            break
    else:
        raise StorageBlocked("Account audit pagination bound reached; report would be incomplete")

    def lineage(file_id):
        seen, pending = set(), [file_id]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(files.get(current, {}).get("parents", []))
        return seen

    totals = {k: 0 for k in ("project_live", "project_trash", "other_live", "other_trash")}
    rows = []
    for item in files.values():
        charged = int(item.get("quotaBytesUsed", item.get("size", 0)))
        ancestors = lineage(item["id"])
        project = project_folder_id is not None and project_folder_id in ancestors
        category = ("project" if project else "other") + (
            "_trash" if item.get("trashed") else "_live"
        )
        totals[category] += charged
        if charged:
            rows.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "category": category,
                    "size_bytes": int(item.get("size", 0)),
                    "charged_bytes": charged,
                    "parents": item.get("parents", []),
                    "ancestor_ids": sorted(ancestors),
                    "ancestor_names": [files[a]["name"] for a in ancestors if a in files],
                    "md5": item.get("md5Checksum"),
                    "trashed": bool(item.get("trashed")),
                    "charged_minus_size_bytes": charged - int(item.get("size", 0)),
                }
            )
    return {
        "project_folder_id": project_folder_id,
        "quota": quota.report(),
        "totals_bytes": totals,
        "owned_metadata_charged_bytes": sum(totals.values()),
        "unreconciled_drive_usage_bytes": quota.drive_usage - sum(totals.values()),
        "largest_files": sorted(rows, key=lambda row: row["charged_bytes"], reverse=True),
        "note": "Other means outside the supplied folder lineage, not pre-existing user data. "
        "Missing ancestry/revisions/shared storage and unit/timing differences require review. "
        "No deletion was performed.",
    }


def find_folder(service, path):
    parent = "root"
    for name in path:
        page = (
            service.files()
            .list(
                q=f"'{quote(parent)}' in parents and name='{quote(name)}' "
                "and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id,name)",
                pageSize=1000,
            )
            .execute()
        )
        matches = page.get("files", [])
        if len(matches) != 1:
            raise StorageBlocked(f"Folder path ambiguous/missing at {name}; supply exact folder ID")
        parent = matches[0]["id"]
    return parent


class DriveBackend:
    def __init__(self, service, run_id, folder_id=None):
        self.service, self.run_id, self.folder_id = service, run_id, folder_id
        if self.folder_id is None:
            page = (
                service.files()
                .list(
                    q="trashed=false and mimeType='application/vnd.google-apps.folder' "
                    f"and appProperties has {{key='app' and value='{APP}'}} "
                    f"and appProperties has {{key='run' and value='{quote(run_id)}'}}",
                    fields="files(id)",
                )
                .execute()
            )
            matches = page.get("files", [])
            if len(matches) > 1:
                raise StorageBlocked(
                    "Multiple managed folders for this run; inspect before continuing"
                )
            if matches:
                self.folder_id = matches[0]["id"]

    def quota(self):
        return Quota.from_api(self.service.about().get(fields="storageQuota").execute())

    def ensure_folder(self):
        if self.folder_id is None:
            item = (
                self.service.files()
                .create(
                    body={
                        "name": f"ArabicSummarizerRecovery-{self.run_id}",
                        "mimeType": "application/vnd.google-apps.folder",
                        "appProperties": {"app": APP, "run": self.run_id},
                    },
                    fields="id",
                )
                .execute()
            )
            self.folder_id = item["id"]

    def items(self):
        if self.folder_id is None:
            return []
        result, token = [], None
        while True:
            page = (
                self.service.files()
                .list(
                    q=f"'{quote(self.folder_id)}' in parents and trashed=false",
                    pageSize=1000,
                    pageToken=token,
                    fields="nextPageToken,files(" + FIELDS + ")",
                )
                .execute()
            )
            result.extend(page.get("files", []))
            token = page.get("nextPageToken")
            if not token:
                return result

    def managed_bytes(self):
        return sum(int(item.get("quotaBytesUsed", item.get("size", 0))) for item in self.items())

    def _owned(self, item):
        properties = item.get("appProperties", {})
        if (
            item.get("trashed")
            or self.folder_id not in item.get("parents", [])
            or properties.get("app") != APP
            or properties.get("run") != self.run_id
        ):
            raise StorageBlocked("Artifact does not belong to this managed folder/run")
        return item

    def stat(self, file_id):
        item = self._owned(self.service.files().get(fileId=file_id, fields=FIELDS).execute())
        return {"id": item["id"], "size": int(item["size"]), "md5": item["md5Checksum"]}

    def upload(self, path, kind, experiment, step):
        from googleapiclient.http import MediaFileUpload

        self.ensure_folder()
        properties = {"app": APP, "run": self.run_id, "kind": kind, "step": str(step)}
        request = self.service.files().create(
            body={
                "name": f"{kind}-{experiment}-{step}-{uuid.uuid4().hex}.tar.zst",
                "parents": [self.folder_id],
                "appProperties": properties,
            },
            media_body=MediaFileUpload(
                str(path),
                mimetype="application/octet-stream",
                chunksize=32 * 1024 * 1024,
                resumable=True,
            ),
            fields=FIELDS,
        )
        result = None
        while result is None:
            progress, result = request.next_chunk(num_retries=3)
            if progress is not None:
                print(f"Uploading {kind}: {progress.progress() * 100:.0f}%", end="\r")
        self._owned(result)
        return self.stat(result["id"])

    def download(self, file_id, path):
        from googleapiclient.http import MediaIoBaseDownload

        self.stat(file_id)
        with path.open("wb") as stream:
            request = MediaIoBaseDownload(
                stream, self.service.files().get_media(fileId=file_id), chunksize=32 * 1024 * 1024
            )
            done = False
            while not done:
                _, done = request.next_chunk(num_retries=3)

    def _index_items(self):
        return [
            self._owned(item)
            for item in self.items()
            if item.get("appProperties", {}).get("kind") == "index"
        ]

    def read_index(self):
        matches = self._index_items()
        if not matches:
            return None
        if len(matches) != 1:
            raise StorageBlocked("Ambiguous managed index; no writes or cleanup allowed")
        content = self.service.files().get_media(fileId=matches[0]["id"]).execute()
        return json.loads(content)

    def write_index(self, index):
        from googleapiclient.http import MediaIoBaseUpload

        self.ensure_folder()
        media = MediaIoBaseUpload(
            io.BytesIO(json.dumps(index, allow_nan=False).encode()),
            mimetype="application/json",
            resumable=False,
        )
        existing = self._index_items()
        if len(existing) > 1:
            raise StorageBlocked("Ambiguous index")
        if existing:
            self.service.files().update(
                fileId=existing[0]["id"], media_body=media, fields="id"
            ).execute()
        else:
            self.service.files().create(
                body={
                    "name": "storage-index.json",
                    "parents": [self.folder_id],
                    "appProperties": {"app": APP, "run": self.run_id, "kind": "index"},
                },
                media_body=media,
                fields="id",
            ).execute()

    def delete_managed(self, artifact, confirmation=None):
        if confirmation != "DELETE " + artifact["id"] + " " + artifact["md5"]:
            raise StorageBlocked("Explicit artifact deletion confirmation required")
        # Called only for an obsolete ID from a read-back-verified index. Never
        # accepts a mounted path, never empties Trash, never touches legacy files.
        try:
            item = self._owned(
                self.service.files().get(fileId=artifact["id"], fields=FIELDS).execute()
            )
        except Exception as exc:
            if getattr(getattr(exc, "resp", None), "status", None) == 404:
                return  # Prior confirmed DELETE completed before an index write was interrupted.
            raise
        if item.get("appProperties", {}).get("kind") not in {"full", "weights"}:
            raise StorageBlocked("Only managed payloads may be retired")
        if self.stat(artifact["id"]) != {k: artifact[k] for k in ("id", "size", "md5")}:
            raise StorageBlocked("Obsolete artifact identity/checksum changed")
        self.service.files().delete(fileId=artifact["id"]).execute()

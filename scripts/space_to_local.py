import os
import io
import re
import gzip
import tarfile
import sqlite3
import tempfile
from pathlib import Path

import boto3
import pandas as pd
from botocore.client import Config
from dotenv import load_dotenv
load_dotenv()


# =========================
# CONFIG
# =========================
SPACE_REGION = "nyc3"
SPACE_NAME = "hyperesearchstore"

RAW_PREFIX = "latent-fair-value/btc/ubuntu-s-1vcpu-512mb-10gb-nyc3-01/"

# local output paths
LOCAL_BASE = Path("data/research-datasets/latent-fair-value/btc")
MANIFEST_PATH = LOCAL_BASE / "_manifest.parquet"

EXPECTED_TABLES = [
    "price_snapshots",
    "asset_context_snapshots",
]

ACCESS_KEY = os.environ["SPACES_KEY"]
SECRET_KEY = os.environ["SPACES_SECRET"]


# =========================
# CLIENT
# =========================
session = boto3.session.Session()
s3 = session.client(
    "s3",
    region_name=SPACE_REGION,
    endpoint_url=f"https://{SPACE_REGION}.digitaloceanspaces.com",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4"),
)


# =========================
# HELPERS
# =========================
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def ensure_dirs() -> None:
    (LOCAL_BASE / "price_snapshots").mkdir(parents=True, exist_ok=True)
    (LOCAL_BASE / "asset_context_snapshots").mkdir(parents=True, exist_ok=True)


def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_parquet(MANIFEST_PATH)

    return pd.DataFrame(columns=[
        "archive_key",
        "etag",
        "last_modified",
        "price_rows_written",
        "asset_context_rows_written",
        "price_parquet_path",
        "asset_context_parquet_path",
        "tables_found",
    ])


def save_manifest(df: pd.DataFrame) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANIFEST_PATH, index=False)


def list_archives():
    out = []
    paginator = s3.get_paginator("list_objects_v2")

    print(f"Listing archives under s3://{SPACE_NAME}/{RAW_PREFIX}")

    for page in paginator.paginate(Bucket=SPACE_NAME, Prefix=RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if (
                key.endswith(".tar.gz")
                or key.endswith(".sqlite3.gz")
                or key.endswith(".sqlite.gz")
                or key.endswith(".db.gz")
            ):
                out.append({
                    "key": key,
                    "etag": obj["ETag"].strip('"'),
                    "last_modified": str(obj["LastModified"]),
                    "size_bytes": obj.get("Size", 0),
                })

    out.sort(key=lambda x: x["key"])
    print(f"Found {len(out)} archive(s)")
    return out


def already_processed(manifest: pd.DataFrame, archive_key: str, etag: str) -> bool:
    if manifest.empty:
        return False
    matches = manifest[
        (manifest["archive_key"] == archive_key) &
        (manifest["etag"] == etag)
    ]
    return not matches.empty


def extract_sqlite_from_tar_gz(archive_path: str) -> str:
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()

        sqlite_member = None
        for member in members:
            lower = member.name.lower()
            if lower.endswith(".sqlite3") or lower.endswith(".sqlite") or lower.endswith(".db"):
                sqlite_member = member
                break

        if sqlite_member is None:
            names = [m.name for m in members]
            raise ValueError(
                f"No sqlite file found inside archive: {archive_path}\n"
                f"Archive members: {names}"
            )

        print(f"  Extracting sqlite member: {sqlite_member.name}")

        extracted = tar.extractfile(sqlite_member)
        if extracted is None:
            raise ValueError(f"Could not extract sqlite member: {sqlite_member.name}")

        tmp_sqlite = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite3")
        tmp_sqlite.write(extracted.read())
        tmp_sqlite.flush()
        tmp_sqlite.close()

        return tmp_sqlite.name


def extract_sqlite_from_gzip(archive_path: str, archive_key: str) -> str:
    suffix = ".sqlite3"
    lower = archive_key.lower()
    if lower.endswith(".sqlite.gz"):
        suffix = ".sqlite"
    elif lower.endswith(".db.gz"):
        suffix = ".db"

    with gzip.open(archive_path, "rb") as gz:
        tmp_sqlite = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_sqlite.write(gz.read())
        tmp_sqlite.flush()
        tmp_sqlite.close()
        return tmp_sqlite.name


def get_existing_tables(conn: sqlite3.Connection) -> list[str]:
    df = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
        """,
        conn,
    )
    return df["name"].tolist()


def read_tables_from_archive(archive_key: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    tmp_suffix = ".tar.gz" if archive_key.endswith(".tar.gz") else ".gz"
    with tempfile.NamedTemporaryFile(suffix=tmp_suffix) as tmp_archive:
        print(f"  Downloading archive from Spaces...")
        s3.download_fileobj(SPACE_NAME, archive_key, tmp_archive)
        tmp_archive.flush()

        if archive_key.endswith(".tar.gz"):
            sqlite_path = extract_sqlite_from_tar_gz(tmp_archive.name)
        else:
            sqlite_path = extract_sqlite_from_gzip(tmp_archive.name, archive_key)

        try:
            conn = sqlite3.connect(sqlite_path)
            try:
                existing_tables = get_existing_tables(conn)
                print(f"  Tables found: {existing_tables}")

                out = {}
                for table in EXPECTED_TABLES:
                    if table not in existing_tables:
                        print(f"  Skipping missing table: {table}")
                        continue

                    print(f"  Reading table: {table}")
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    df["source_archive_key"] = archive_key
                    out[table] = df
                    print(f"    Loaded {len(df):,} rows from {table}")

                return out, existing_tables
            finally:
                conn.close()
        finally:
            try:
                os.remove(sqlite_path)
            except OSError:
                pass


def write_local_parquet(df: pd.DataFrame, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local_path, index=False)


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()
    manifest = load_manifest()
    archives = list_archives()

    if not archives:
        print("No archives found.")
        return

    print(f"Manifest currently has {len(manifest)} processed archive record(s)")
    new_entries = []

    total = len(archives)
    for i, obj in enumerate(archives, start=1):
        archive_key = obj["key"]
        etag = obj["etag"]

        print()
        print("=" * 80)
        print(f"[{i}/{total}] Processing: {archive_key}")
        print("=" * 80)

        if already_processed(manifest, archive_key, etag):
            print("Already processed with same ETag, skipping.")
            continue

        try:
            tables, existing_tables = read_tables_from_archive(archive_key)
        except Exception as e:
            print(f"FAILED on {archive_key}")
            print(f"Error: {e}")
            continue

        archive_name = Path(archive_key).name
        if archive_name.endswith(".tar.gz"):
            archive_stem = archive_name[:-7]
        elif archive_name.endswith(".sqlite3.gz"):
            archive_stem = archive_name[:-11]
        elif archive_name.endswith(".sqlite.gz"):
            archive_stem = archive_name[:-10]
        elif archive_name.endswith(".db.gz"):
            archive_stem = archive_name[:-6]
        else:
            archive_stem = Path(archive_name).stem

        price_rows = 0
        context_rows = 0
        price_path = None
        context_path = None

        if "price_snapshots" in tables:
            price_df = tables["price_snapshots"]
            price_df["source_etag"] = etag

            price_path = (
                LOCAL_BASE
                / "price_snapshots"
                / f"part-{safe_name(archive_stem)}-{etag}.parquet"
            )

            print(f"  Writing price_snapshots -> {price_path}")
            write_local_parquet(price_df, price_path)
            price_rows = len(price_df)

        if "asset_context_snapshots" in tables:
            context_df = tables["asset_context_snapshots"]
            context_df["source_etag"] = etag

            context_path = (
                LOCAL_BASE
                / "asset_context_snapshots"
                / f"part-{safe_name(archive_stem)}-{etag}.parquet"
            )

            print(f"  Writing asset_context_snapshots -> {context_path}")
            write_local_parquet(context_df, context_path)
            context_rows = len(context_df)

        new_entries.append({
            "archive_key": archive_key,
            "etag": etag,
            "last_modified": obj["last_modified"],
            "price_rows_written": price_rows,
            "asset_context_rows_written": context_rows,
            "price_parquet_path": str(price_path) if price_path else None,
            "asset_context_parquet_path": str(context_path) if context_path else None,
            "tables_found": ",".join(existing_tables),
        })

        manifest = pd.concat([manifest, pd.DataFrame([new_entries[-1]])], ignore_index=True)
        save_manifest(manifest)

        print("  Done.")
        print(f"  price rows written: {price_rows:,}")
        print(f"  context rows written: {context_rows:,}")
        print(f"  manifest saved: {MANIFEST_PATH}")

    print()
    print("#" * 80)
    print("Finished run.")
    print(f"Manifest path: {MANIFEST_PATH}")
    print(f"Output base:   {LOCAL_BASE}")
    print("#" * 80)


if __name__ == "__main__":
    main()

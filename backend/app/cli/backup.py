import argparse
import sys
from pathlib import Path

from app.application.backup_service import (
    BackupError,
    create_backup,
    manifest_payload,
)
from app.config import get_settings


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive the SQLite database, the paper library and the vector index "
            "into one file. The archive carries a manifest so a restore can be "
            "verified."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Archive path to write; must not already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    settings = get_settings()
    try:
        result = create_backup(settings, args.output)
    except BackupError as error:
        print(f"backup failed: {error}", file=sys.stderr)
        return 1
    print(manifest_payload(result.manifest))
    print(f"archive: {result.archive} ({result.bytes_written} bytes)")
    if not result.manifest.database_included or not result.manifest.papers_included:
        print(
            "warning: the archive is incomplete; check the manifest notes above",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

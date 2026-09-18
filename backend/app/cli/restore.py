import argparse
import sys
from pathlib import Path

from app.application.backup_service import BackupError, restore_backup
from app.config import get_settings


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a PaperPilot backup into the configured data locations and "
            "verify the result. This replaces the database, paper library and "
            "vector index, so it refuses to run without --force."
        )
    )
    parser.add_argument("archive", type=Path, help="Archive produced by app.cli.backup")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Confirm that the current database, papers and vectors may be replaced",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    settings = get_settings()
    try:
        verification = restore_backup(settings, args.archive, force=args.force)
    except BackupError as error:
        print(f"restore failed: {error}", file=sys.stderr)
        return 1

    print(
        f"restored {verification.restored_papers} paper manifest(s), "
        f"database={'yes' if verification.restored_database else 'no'}, "
        f"vectors={'yes' if verification.restored_vectors else 'no'}"
    )
    print(f"archive created at: {verification.manifest.created_at.isoformat()}")
    print(f"index fingerprint in archive: {verification.manifest.index_signature or 'unknown'}")
    for warning in verification.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not verification.consistent:
        print(
            "restore verification failed; do not trust this library until it is "
            "re-checked",
            file=sys.stderr,
        )
        return 2
    print("restore verified; next check: curl -s http://localhost:8000/api/v1/health")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

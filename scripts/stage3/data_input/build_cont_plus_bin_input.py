#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from common import resolve_repo_path  # noqa: E402
from data_input.shared import run_mode_build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage3 Part I cont_plus_bin feature input view.")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.yaml")
    parser.add_argument(
        "--no-legacy-view",
        action="store_true",
        help="Do not write legacy interaction_feature_view.csv alias.",
    )
    args = parser.parse_args()

    repo_root = SCRIPT_DIR.parents[2]
    manifest_path = resolve_repo_path(repo_root, args.manifest)

    summary = run_mode_build(
        repo_root=repo_root,
        manifest_path=manifest_path,
        feature_mode="cont_plus_bin",
        write_legacy_view=not args.no_legacy_view,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

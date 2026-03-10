#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from data_input.shared import run_mode_build
from common import repo_root_from_file, resolve_repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 Part I - Data Prep And Feature Derivation (cont_plus_bin compatibility entry)")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.yaml")
    args = parser.parse_args()

    repo_root = repo_root_from_file(__file__)
    manifest_path = resolve_repo_path(repo_root, args.manifest)

    summary = run_mode_build(
        repo_root=repo_root,
        manifest_path=manifest_path,
        feature_mode="cont_plus_bin",
        write_legacy_view=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

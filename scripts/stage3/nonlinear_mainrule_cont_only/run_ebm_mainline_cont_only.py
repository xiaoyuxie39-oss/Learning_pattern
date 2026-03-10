#!/usr/bin/env python3
from __future__ import annotations

from v2_run_model import run_job_cli


if __name__ == "__main__":
    run_job_cli(model_name="ebm", branch_name="mainline")

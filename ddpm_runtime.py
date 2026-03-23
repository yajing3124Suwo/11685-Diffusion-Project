# -*- coding: utf-8 -*-
"""
Runtime profiles for local machine, Google Colab, and PSC Bridges-2 (batch/interactive).

Environment variables (optional):
  DDPM_RUNTIME   colab | psc | local  — overrides --runtime when set
  DDPM_DATA_DIR  — override args.data_dir (e.g. Ocean scratch)
  DDPM_OUTPUT_DIR — override args.output_dir (e.g. Ocean project directory)
"""
import os
import sys


def resolve_runtime(cli_runtime):
    """
    Decide which profile to use.
    Precedence: DDPM_RUNTIME env, then non-auto CLI value, then Colab detection, else local.
    """
    env_r = os.environ.get("DDPM_RUNTIME", "").strip().lower()
    if env_r in ("colab", "psc", "local"):
        return env_r
    if cli_runtime and cli_runtime != "auto":
        return cli_runtime
    if _detect_colab():
        return "colab"
    return "local"


def _detect_colab():
    """True when running inside Google Colab (optional 'google.colab' package)."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def apply_runtime_to_args(args):
    """Mutate argparse namespace for Colab/PSC paths and DataLoader settings."""
    r = resolve_runtime(getattr(args, "runtime", "auto"))
    args._ddpm_runtime = r

    data_override = os.environ.get("DDPM_DATA_DIR")
    if data_override:
        args.data_dir = data_override

    out_override = os.environ.get("DDPM_OUTPUT_DIR")
    if out_override:
        args.output_dir = out_override

    if r == "colab":
        # Multiprocessing DataLoader workers often break or hang in Colab notebooks.
        args.num_workers = 0

    return args

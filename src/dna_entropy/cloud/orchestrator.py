"""Drive the user's own gcloud to run Evo on the always-on GPU VM in THEIR project.

The VM is provisioned and kept alive by the separate keeper (``keep_gpu.py`` /
``dna-entropy keep-gpu``). This module only *uses* it: find the box, wake it if stopped,
ensure the Evo stack, upload the locus, run, download to Downloads\\<name>\\, and LEAVE
THE BOX RUNNING. It never creates, stops, or deletes a VM.

The provisioning helpers (``_create_box`` and the zone/offer escalation ``_try_*``) live
here because the keeper reuses them; ``run_in_cloud`` itself does not call them.
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer

from . import gcloud
from .ui import Spinner

# A single, stable box name per project so we can detect & reuse a saved one.
BOX_NAME = "dna-entropy-box"

# Public Google Deep Learning image (torch 2.9 + CUDA 12.9 + driver) — no hosting by us.
BASE_IMAGE_FAMILY = "pytorch-2-9-cu129-ubuntu-2404-nvidia-580"
BASE_IMAGE_PROJECT = "deeplearning-platform-release"

# GPU choices to try, cheapest first; A100 is the stockout fallback (different pool).
# L4 quota is granted by default on new projects; A100 requires a manual quota request.
OFFERS = [
    ("g2-standard-8", "nvidia-l4", "L4"),
    ("a2-highgpu-1g", "nvidia-tesla-a100", "A100"),
]
DEFAULT_ZONES = [
    # US
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "us-east1-b", "us-east1-c", "us-east1-d",
    "us-east4-a", "us-east4-b", "us-east4-c",
    "us-west1-a", "us-west1-b", "us-west1-c",
    "us-west4-a", "us-west4-b",
    "us-south1-a", "us-south1-b", "us-south1-c",
    # Europe
    "europe-west1-b", "europe-west1-c", "europe-west1-d",
    "europe-west2-a", "europe-west2-b", "europe-west2-c",
    "europe-west3-a", "europe-west3-b", "europe-west3-c",
    "europe-west4-a", "europe-west4-b", "europe-west4-c",
    "europe-west6-a", "europe-west6-b", "europe-west6-c",
    "europe-north1-a", "europe-north1-b", "europe-north1-c",
    "europe-central2-a", "europe-central2-b", "europe-central2-c",
    # Asia-Pacific
    "asia-east1-a", "asia-east1-b", "asia-east1-c",
    "asia-east2-a", "asia-east2-b", "asia-east2-c",
    "asia-northeast1-a", "asia-northeast1-b", "asia-northeast1-c",
    "asia-northeast3-a", "asia-northeast3-b", "asia-northeast3-c",
    "asia-south1-a", "asia-south1-b", "asia-south1-c",
    "asia-south2-a", "asia-south2-b", "asia-south2-c",
    "asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c",
    "asia-southeast2-a", "asia-southeast2-b", "asia-southeast2-c",
    "australia-southeast1-a", "australia-southeast1-b", "australia-southeast1-c",
    # Middle East / Americas
    "me-central1-a", "me-central1-b", "me-central1-c",
    "me-west1-a", "me-west1-b", "me-west1-c",
    "southamerica-east1-a", "southamerica-east1-b", "southamerica-east1-c",
    "southamerica-west1-a", "southamerica-west1-b", "southamerica-west1-c",
    "northamerica-northeast1-a", "northamerica-northeast1-b", "northamerica-northeast1-c",
]

# Installed on a fresh VM, from public sources only. Idempotent (fast on a reused box).
_VM_SETUP_SCRIPT = """#!/usr/bin/env bash
set -e
if python3 -c 'import evo2, flash_attn, Bio' 2>/dev/null; then echo EVO_STACK_PRESENT; exit 0; fi
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | head -1 || echo /usr/local/cuda)"
ARCH="$(python3 -c 'import torch;cc=torch.cuda.get_device_capability();print(f"{cc[0]}.{cc[1]}")')"
export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS=4
python3 -m pip install --break-system-packages -q typer biopython pyrodigal evo2 ninja
python3 -m pip install --break-system-packages --no-build-isolation flash-attn==2.8.3
python3 -c 'import evo2, flash_attn; print("EVO_STACK_OK")'
"""

LLM_HINT = (
    "Stuck? Copy the error above into an LLM (Claude / ChatGPT) and ask how to fix it - "
    "these are usually quick to resolve."
)

_INSTALL_MSG = """gcloud (the Google Cloud CLI) is not installed.
  1. Install it:  https://cloud.google.com/sdk/docs/install
  2. New terminal, then run:  gcloud auth login
  3. Set your project:        gcloud config set project YOUR_PROJECT_ID
  4. Re-run this app."""

_AUTH_MSG = """You are not signed in to Google Cloud.
  Run:  gcloud auth login
  Then: gcloud config set project YOUR_PROJECT_ID"""

_PROJECT_MSG = """No Google Cloud project is set.
  Create one at https://console.cloud.google.com/projectcreate
  Then run:  gcloud config set project YOUR_PROJECT_ID"""

def _quota_msg(label: str, accel: str) -> str:
    """Return a quota error message specific to the GPU type that was denied."""
    if "l4" in accel.lower():
        metric = "NVIDIA_L4_GPUS"
    else:
        metric = "NVIDIA_A100_GPUS"
    return (
        f"Your project has no {label} GPU quota yet - this is a one-time request.\n"
        "  1. Open: https://console.cloud.google.com/iam-admin/quotas\n"
        "  2. Filter by:\n"
        "       Service:  Compute Engine API\n"
        f"       Metric:   {metric}\n"
        "     Tick a zone in YOUR region (e.g. us-central1 for US, europe-west4 for EU),\n"
        "     then click EDIT QUOTAS.\n"
        "  3. Request a limit of 1 and submit. L4 quota is usually auto-approved in minutes.\n"
        "  4. Re-run this app once it is approved."
    )


@dataclass
class CloudConfig:
    project: Optional[str] = None
    image_family: str = BASE_IMAGE_FAMILY
    image_project: str = BASE_IMAGE_PROJECT
    offers: list = field(default_factory=lambda: list(OFFERS))
    zones: list = field(default_factory=lambda: list(DEFAULT_ZONES))
    boot_disk_gb: int = 100
    ssh_key_file: Optional[str] = None


# --- small persistent state (remember last-good zone, ssh key) ----------------------

def _state_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "dna-entropy" / "config.json"


def load_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _package_dir() -> Path:
    """Path to the `dna_entropy` source to upload to the VM.

    In the frozen .exe the source is bundled via `--add-data src/dna_entropy;_pkgsrc`.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", ".")) / "_pkgsrc"
    return Path(__file__).resolve().parent.parent


# --- preflight ----------------------------------------------------------------------

def preflight(cfg: CloudConfig) -> tuple[str, str]:
    if gcloud.find_gcloud() is None:
        raise gcloud.GcloudNotInstalled(_INSTALL_MSG)
    account = gcloud.active_account()
    if not account:
        raise gcloud.GcloudNotAuthenticated(_AUTH_MSG)
    project = cfg.project or gcloud.get_project()
    if not project:
        raise gcloud.GcloudError(_PROJECT_MSG)
    return account, project


def _ordered_zones(cfg: CloudConfig, state: dict) -> list[str]:
    last = state.get("last_zone")
    zones = list(cfg.zones)
    if last and last in zones:
        zones.remove(last)
        zones.insert(0, last)
    return zones


# Create failures that are project-wide setup problems, not zone-specific. Retrying other
# zones is pointless — every zone fails identically — so we surface these immediately with
# the exact gcloud text plus how to fix it.
_SETUP_ERROR_KINDS = ("billing", "api_disabled", "permission")


def _setup_remediation(kind: str, raw: str) -> str:
    """Human-fixable remediation text for a project-wide setup error, incl. the raw error."""
    if kind == "billing":
        return (
            "Billing is not enabled on this project - GCP will not create ANY VM without it.\n"
            f"  gcloud said: {raw}\n"
            "  Fix: link a billing account at https://console.cloud.google.com/billing\n"
            "       or: gcloud billing projects link PROJECT_ID "
            "--billing-account=XXXXXX-XXXXXX-XXXXXX"
        )
    if kind == "api_disabled":
        return (
            "The Compute Engine API is not enabled on this project.\n"
            f"  gcloud said: {raw}\n"
            "  Fix: gcloud services enable compute.googleapis.com\n"
            "       (or enable it at "
            "https://console.cloud.google.com/apis/library/compute.googleapis.com )"
        )
    return (
        "Your account is not allowed to create VMs in this project.\n"
        f"  gcloud said: {raw}\n"
        "  Fix: have the project owner grant your account 'Compute Admin' "
        "(roles/compute.admin),\n"
        "       or switch to an account that has it (gcloud auth login)."
    )


def _raise_if_setup(errors: dict[str, str]) -> None:
    """Raise a remediation error if any project-wide setup failure has been seen."""
    for kind in _SETUP_ERROR_KINDS:
        if kind in errors:
            raise gcloud.GcloudError(_setup_remediation(kind, errors[kind]))


def _format_errors(errors: dict[str, str]) -> str:
    """One exact-gcloud-error example per failure kind, for the final give-up message."""
    if not errors:
        return ""
    lines = ["  Exact errors gcloud returned (one example per kind):"]
    for kind, raw in errors.items():
        first = next((ln for ln in raw.strip().splitlines() if ln.strip()), raw)
        lines.append(f"    [{kind}] {first.strip()}")
    return "\n".join(lines) + "\n"


def _attempt_create(zone: str, machine: str, accel: str, project: str, cfg: CloudConfig) -> None:
    """Create BOX_NAME in one zone; raises GcloudError on failure."""
    gcloud.create_vm(
        BOX_NAME, zone,
        machine_type=machine, accelerator=accel,
        image_family=cfg.image_family, image_project=cfg.image_project,
        boot_disk_gb=cfg.boot_disk_gb, project=project,
    )


def _try_sequential(zones: list[str], machine: str, accel: str, label: str,
                    project: str, cfg: CloudConfig,
                    errors: dict[str, str]) -> tuple[Optional[str], bool, bool]:
    """Try zones one at a time.

    Returns (winning_zone, had_quota_error, was_pre_existing). Records one exact error
    per kind into ``errors``. Raises immediately (with remediation) on a project-wide
    setup problem — billing/API/permission — since retrying other zones cannot help.
    """
    had_quota = False
    for zone in zones:
        try:
            with Spinner(f"[create] Starting a {label} VM in {zone}"):
                _attempt_create(zone, machine, accel, project, cfg)
            return zone, had_quota, False
        except gcloud.GcloudError as exc:
            raw = str(exc)
            kind = gcloud.classify_create_error(raw)
            errors[kind] = raw
            if kind in _SETUP_ERROR_KINDS:
                raise gcloud.GcloudError(_setup_remediation(kind, raw))
            if kind == "already_exists":
                typer.secho(f"      {zone}: box already exists here - reusing it", fg=typer.colors.CYAN)
                return zone, had_quota, True
            if kind == "quota":
                had_quota = True
                typer.secho(f"      {zone}: no {label} quota in this region, continuing...", fg=typer.colors.YELLOW)
            else:
                typer.secho(f"      {zone}: no {label} capacity, trying next...", fg=typer.colors.YELLOW)
    return None, had_quota, False


def _try_parallel(zones: list[str], machine: str, accel: str, label: str,
                  project: str, cfg: CloudConfig,
                  errors: dict[str, str]) -> tuple[Optional[str], bool, bool]:
    """Try zones simultaneously; return (first_winner, had_quota_error, was_pre_existing).

    Cleans up any extra VMs that also succeeded. Records one exact error per kind into
    ``errors``. Does not raise on setup errors itself (workers can't raise cleanly) — the
    caller inspects ``errors`` via :func:`_raise_if_setup` after the batch.
    """
    typer.echo(f"  Trying {len(zones)} zones in parallel: {', '.join(zones)}")

    lock = threading.Lock()
    winner: list[Optional[str]] = [None]
    had_quota: list[bool] = [False]
    pre_existing: list[bool] = [False]

    def attempt(zone: str) -> tuple:
        try:
            _attempt_create(zone, machine, accel, project, cfg)
            with lock:
                if winner[0] is None:
                    winner[0] = zone
                    return ("ok", zone, False)
                return ("extra", zone, False)
        except gcloud.GcloudError as exc:
            raw = str(exc)
            kind = gcloud.classify_create_error(raw)
            with lock:
                errors[kind] = raw
                if kind == "quota":
                    had_quota[0] = True
            if kind == "already_exists":
                with lock:
                    if winner[0] is None:
                        winner[0] = zone
                        pre_existing[0] = True
                        return ("ok", zone, True)
                    return ("extra", zone, True)
            return ("fail", zone, kind)

    with ThreadPoolExecutor(max_workers=len(zones)) as executor:
        results = list(executor.map(attempt, zones))

    for result in results:
        if result[0] == "ok":
            label_str = "box already exists here - reusing" if result[2] else f"{label} VM ready!"
            typer.secho(f"      {result[1]}: {label_str}", fg=typer.colors.CYAN if result[2] else typer.colors.GREEN)
        elif result[0] == "extra":
            if not result[2]:  # don't delete a pre-existing VM we didn't create
                typer.secho(f"      {result[1]}: also came up - removing duplicate", fg=typer.colors.YELLOW)
                try:
                    gcloud.delete_vm(BOX_NAME, result[1], project=project)
                except Exception:
                    pass
        else:
            _, zone, kind = result
            if kind == "quota":
                typer.secho(f"      {zone}: no {label} quota in this region", fg=typer.colors.YELLOW)
            else:
                typer.secho(f"      {zone}: no {label} capacity", fg=typer.colors.YELLOW)

    return winner[0], had_quota[0], pre_existing[0]


def _create_box(project: str, state: dict, cfg: CloudConfig) -> tuple[str, bool]:
    """Create BOX_NAME across zones with escalating parallelism.

    Returns (zone, created) where created=False means a pre-existing VM was found.
    Per GPU tier: 3 sequential -> 3 parallel -> 5 parallel -> 7 parallel (repeat).
    Quota errors in individual regions do NOT abort — we keep trying other regions.
    Only escalates to A100 after all L4 zones are exhausted.
    """
    errors: dict[str, str] = {}  # one exact gcloud error per kind, for diagnostics
    for machine, accel, label in cfg.offers:
        zones = _ordered_zones(cfg, state)
        any_quota = False

        def _win(zone: str, created: bool) -> tuple[str, bool]:
            state["last_zone"] = zone
            save_state(state)
            return zone, created

        # Phase 1: first 3 sequential (fast feedback, common case)
        seq, zones = zones[:3], zones[3:]
        won, q, existing = _try_sequential(seq, machine, accel, label, project, cfg, errors)
        any_quota = any_quota or q
        if won:
            return _win(won, not existing)

        # Phase 2: next 3 parallel
        if zones:
            batch, zones = zones[:3], zones[3:]
            typer.secho("  Sequential attempts exhausted; switching to parallel...", fg=typer.colors.YELLOW)
            won, q, existing = _try_parallel(batch, machine, accel, label, project, cfg, errors)
            any_quota = any_quota or q
            if won:
                return _win(won, not existing)
        _raise_if_setup(errors)

        # Phase 3: next 5 parallel
        if zones:
            batch, zones = zones[:5], zones[5:]
            won, q, existing = _try_parallel(batch, machine, accel, label, project, cfg, errors)
            any_quota = any_quota or q
            if won:
                return _win(won, not existing)
        _raise_if_setup(errors)

        # Phase 4+: batches of 7 until all zones are tried
        while zones:
            batch, zones = zones[:7], zones[7:]
            won, q, existing = _try_parallel(batch, machine, accel, label, project, cfg, errors)
            any_quota = any_quota or q
            if won:
                return _win(won, not existing)
        _raise_if_setup(errors)

        if any_quota:
            typer.secho(
                f"  NOTE: some regions had no {label} quota. "
                "If you have quota, those regions will work on retry.",
                fg=typer.colors.YELLOW,
            )
        typer.secho(f"  {label} unavailable in all zones; trying a bigger GPU...", fg=typer.colors.YELLOW)

    raise gcloud.GcloudError(
        "No GPU capacity found in any zone right now.\n"
        + _format_errors(errors)
        + "  - For L4: stockout is common; try again in a few minutes.\n"
        f"  - If you saw quota warnings above:\n{_quota_msg('L4', 'nvidia-l4')}"
    )


def _wait_for_ssh(box: str, zone: str, cfg: CloudConfig, project: str) -> None:
    deadline = time.monotonic() + 240
    with Spinner("[connect] Waiting for the VM to accept connections"):
        while True:
            proc = gcloud.ssh(box, zone, "echo ready", project=project,
                              key_file=cfg.ssh_key_file, timeout=60, check=False)
            if proc.returncode == 0 and "ready" in proc.stdout:
                return
            if time.monotonic() > deadline:
                raise gcloud.GcloudError("VM never became reachable over SSH (timed out).")
            time.sleep(5)


def _ensure_evo(box: str, zone: str, cfg: CloudConfig, project: str) -> None:
    tmpdir = Path(tempfile.gettempdir())
    setup = tmpdir / "dna_entropy_vm_setup.sh"
    setup.write_text(_VM_SETUP_SCRIPT, encoding="utf-8", newline="\n")
    # Pack the package into one tar.gz — single-file scp is reliable (recursive pscp isn't).
    tar_path = tmpdir / "dna_entropy_pkg.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(_package_dir(), arcname="dna_entropy")
    with Spinner("[setup] Uploading the tool"):
        gcloud.scp(str(setup), f"{box}:vm_setup.sh", zone, project=project, key_file=cfg.ssh_key_file, timeout=120)
        gcloud.scp(str(tar_path), f"{box}:pkg.tar.gz", zone, project=project, key_file=cfg.ssh_key_file, timeout=180)
    with Spinner("[setup] Installing Evo (first time on a new box ~10 min; instant when reused)"):
        gcloud.ssh(
            box, zone,
            "rm -rf ~/dna_entropy && tar xzf ~/pkg.tar.gz -C ~ && bash ~/vm_setup.sh",
            project=project, key_file=cfg.ssh_key_file, timeout=1800,
        )


def _no_box_error() -> gcloud.GcloudError:
    """The box is provisioned by the always-on keeper, not by this app."""
    return gcloud.GcloudError(
        "No GPU box found in your project.\n"
        "  This app no longer creates VMs - the always-on keeper does. Start it first:\n"
        "      python keep_gpu.py            (or: dna-entropy keep-gpu)\n"
        "  Leave that window open until the box is UP, then re-run this."
    )


def run_in_cloud(
    *, seq: str, name: str, base_dir: Path, genes: bool, cfg: CloudConfig,
    input_file: Optional[str] = None,
) -> Path:
    """Run Evo on the keeper's always-on GPU box, then LEAVE IT RUNNING.

    This app never creates or tears down a VM. It requires a box provisioned by the keeper
    (``keep_gpu.py``); it may wake a stopped box, but never stops or deletes one.

    If ``input_file`` is given (GenBank/FASTA), that file is uploaded verbatim (extension
    preserved) so the remote pipeline auto-detects its format and, for GenBank, uses its
    existing genes. Otherwise the validated ``seq`` is uploaded as a plain locus file.
    """
    account, project = preflight(cfg)
    typer.echo(f"  account: {account}")
    typer.echo(f"  project: {project}")
    state = load_state()
    if cfg.ssh_key_file is None:
        cfg.ssh_key_file = state.get("ssh_key_file")

    existing = gcloud.find_instance(BOX_NAME, project)
    if not existing:
        raise _no_box_error()
    zone, status = existing
    typer.secho(f"  Found your GPU box in {zone} ({status or 'starting'}).", fg=typer.colors.CYAN)
    if status in ("TERMINATED", "STOPPED", "SUSPENDED"):
        with Spinner(f"[start] Waking your box in {zone}"):
            gcloud.start_vm(BOX_NAME, zone, project=project)
    # RUNNING / STAGING / PROVISIONING / blank (transitional) — leave it alone.

    _wait_for_ssh(BOX_NAME, zone, cfg, project)
    _ensure_evo(BOX_NAME, zone, cfg, project)

    if input_file:
        # Preserve the extension so the remote pipeline auto-detects GenBank/FASTA.
        ext = Path(input_file).suffix or ".txt"
        remote_name = f"input{ext}"
        local_src = input_file
    else:
        remote_name = "locus.txt"
        tmp = Path(tempfile.gettempdir()) / "dna_entropy_locus.txt"
        tmp.write_text(seq + "\n", encoding="utf-8", newline="\n")
        local_src = str(tmp)
    with Spinner("[upload] Uploading your input"):
        gcloud.scp(local_src, f"{BOX_NAME}:{remote_name}", zone, project=project,
                   key_file=cfg.ssh_key_file, timeout=120)

    genes_flag = "--genes" if genes else "--no-genes"
    remote = (
        f"PYTHONPATH=$HOME python3 -m dna_entropy.cli run -i $HOME/{remote_name} "
        f"--name {name} --predictor evo {genes_flag} --out $HOME/runs"
    )
    with Spinner("[run] Running Evo 2 on the GPU"):
        proc = gcloud.ssh(BOX_NAME, zone, remote, project=project,
                          key_file=cfg.ssh_key_file, timeout=1800)
    for line in proc.stdout.strip().splitlines()[-8:]:
        typer.echo(f"      {line}")

    base_dir.mkdir(parents=True, exist_ok=True)
    with Spinner("[download] Downloading results"):
        gcloud.scp(f"{BOX_NAME}:runs/{name}", str(base_dir), zone, recurse=True,
                   project=project, key_file=cfg.ssh_key_file, timeout=300)

    typer.secho(
        f"  Box left RUNNING in {zone} for the next run. Delete it yourself when finished:\n"
        f"    gcloud compute instances delete {BOX_NAME} --zone={zone}",
        fg=typer.colors.YELLOW,
    )
    return base_dir / name

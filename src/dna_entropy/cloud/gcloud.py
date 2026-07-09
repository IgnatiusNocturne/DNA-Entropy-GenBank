"""Thin wrappers around the user's own ``gcloud`` CLI.

This and ``orchestrator.py`` are the only cloud-aware modules. We shell out to the user's
authenticated gcloud (no embedded credentials), so subprocess passes argv straight to
gcloud — no shell-quoting pitfalls.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional, Sequence


class GcloudError(RuntimeError):
    """A gcloud command failed."""


class GcloudNotInstalled(GcloudError):
    """gcloud CLI is not on PATH."""


class GcloudNotAuthenticated(GcloudError):
    """No active gcloud account."""


def find_gcloud() -> Optional[str]:
    """Return the path to gcloud, or None if not installed."""
    return shutil.which("gcloud") or shutil.which("gcloud.cmd")


def _run(
    args: Sequence[str],
    *,
    timeout: Optional[float] = None,
    check: bool = True,
    stdin_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    exe = find_gcloud()
    if exe is None:
        raise GcloudNotInstalled("gcloud CLI not found on PATH")
    proc = subprocess.run(
        [exe, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # gcloud/pip output isn't always cp1252-decodable on Windows
        timeout=timeout,
        input=stdin_text,
    )
    if check and proc.returncode != 0:
        raise GcloudError((proc.stderr or proc.stdout).strip())
    return proc


# --- account / project --------------------------------------------------------------

def active_account() -> Optional[str]:
    proc = _run(
        ["auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        check=False,
    )
    return proc.stdout.strip() or None


def get_project() -> Optional[str]:
    proc = _run(["config", "get-value", "project"], check=False)
    p = proc.stdout.strip()
    return p if p and p.lower() != "(unset)" else None


# --- compute ------------------------------------------------------------------------

def create_vm(
    name: str,
    zone: str,
    *,
    machine_type: str,
    accelerator: str,
    image_family: str,
    image_project: str,
    boot_disk_gb: int = 100,
    project: Optional[str] = None,
    timeout: float = 600,
) -> None:
    """Create a GPU VM from a PUBLIC image family. Raises GcloudError on failure."""
    args = [
        "compute", "instances", "create", name,
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--accelerator=type={accelerator},count=1",
        f"--image-family={image_family}",
        f"--image-project={image_project}",
        f"--boot-disk-size={boot_disk_gb}GB",
        "--boot-disk-type=pd-balanced",
        "--maintenance-policy=TERMINATE",
        "--restart-on-failure",
    ]
    if project:
        args.append(f"--project={project}")
    _run(args, timeout=timeout, check=True)


def find_instance(name: str, project: Optional[str] = None) -> Optional[tuple[str, str]]:
    """Return (zone, status) of the named instance if it exists, else None.

    Handles blank status (transitional VM state) and multiple results (parallel
    creation left duplicates) by returning the first RUNNING result, or the first
    result of any kind if none are RUNNING.
    """
    args = [
        "compute", "instances", "list",
        f"--filter=name={name}",
        "--format=value(zone.basename(),status)",
    ]
    if project:
        args.append(f"--project={project}")
    out = _run(args, check=False).stdout.strip()
    if not out:
        return None
    candidates: list[tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            candidates.append((parts[0], parts[1]))
        elif len(parts) == 1:
            candidates.append((parts[0], ""))  # zone known, status blank (transitional)
    if not candidates:
        return None
    # Prefer a RUNNING instance if there are multiple (parallel batch left duplicates).
    for c in candidates:
        if c[1] == "RUNNING":
            return c
    return candidates[0]


def start_vm(name: str, zone: str, *, project: Optional[str] = None, timeout: float = 300) -> None:
    args = ["compute", "instances", "start", name, f"--zone={zone}"]
    if project:
        args.append(f"--project={project}")
    _run(args, timeout=timeout, check=True)


def stop_vm(name: str, zone: str, *, project: Optional[str] = None, timeout: float = 300) -> None:
    args = ["compute", "instances", "stop", name, f"--zone={zone}"]
    if project:
        args.append(f"--project={project}")
    _run(args, timeout=timeout, check=True)


def ssh(
    name: str,
    zone: str,
    command: str,
    *,
    project: Optional[str] = None,
    key_file: Optional[str] = None,
    timeout: Optional[float] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``command`` on the VM. Auto-accepts the host-key prompt (fresh VM each run)."""
    args = ["compute", "ssh", name, f"--zone={zone}", "--command", command, "--quiet"]
    if project:
        args.append(f"--project={project}")
    if key_file:
        args.append(f"--ssh-key-file={key_file}")
    return _run(args, timeout=timeout, check=check, stdin_text="y\n")


def scp(
    src: str,
    dst: str,
    zone: str,
    *,
    recurse: bool = False,
    project: Optional[str] = None,
    key_file: Optional[str] = None,
    timeout: Optional[float] = None,
) -> None:
    args = ["compute", "scp"]
    if recurse:
        args.append("--recurse")
    args += [src, dst, f"--zone={zone}", "--quiet"]
    if project:
        args.append(f"--project={project}")
    if key_file:
        args.append(f"--ssh-key-file={key_file}")
    _run(args, timeout=timeout, check=True, stdin_text="y\n")


def delete_vm(
    name: str, zone: str, *, project: Optional[str] = None, timeout: float = 300
) -> None:
    args = ["compute", "instances", "delete", name, f"--zone={zone}", "--quiet"]
    if project:
        args.append(f"--project={project}")
    _run(args, timeout=timeout, check=True)


def classify_create_error(stderr: str) -> str:
    """Bucket a create failure: quota|stockout|already_exists|permission|other."""
    s = stderr.lower()
    if "quota" in s:
        return "quota"
    if (
        "stockout" in s
        or "zone_resource_pool_exhausted" in s
        or "does not have enough resources" in s
    ):
        return "stockout"
    if "already exists" in s or "resource already exists" in s:
        return "already_exists"
    if "permission" in s or "forbidden" in s or "not authorized" in s:
        return "permission"
    return "other"

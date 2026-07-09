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
    """Bucket a create failure.

    Returns one of: ``billing`` | ``api_disabled`` | ``quota`` | ``stockout`` |
    ``already_exists`` | ``permission`` | ``network`` | ``other``.

    The ``billing`` and ``api_disabled`` buckets matter a lot: they are *project-wide*
    setup problems (not zone-specific), and GCP reports them on the very first create.
    Without singling them out they read as generic "no capacity", which is exactly the
    misleading picture a mis-set-up project produces — creates fail in every zone for a
    reason that has nothing to do with capacity.
    """
    s = stderr.lower()
    # Billing must be checked before quota: some billing errors also mention "account".
    if "billing" in s and ("enable" in s or "disabled" in s or "not found" in s
                            or "not active" in s or "account" in s):
        return "billing"
    if (
        "has not been used in project" in s
        or "accessnotconfigured" in s
        or "serviceusage" in s
        or ("compute" in s and "api" in s and ("disabled" in s or "not enabled" in s))
        or "it is disabled" in s
    ):
        return "api_disabled"
    if "quota" in s:
        return "quota"
    if (
        "stockout" in s
        or "zone_resource_pool_exhausted" in s
        or "does not have enough resources" in s
        or "resource_availability" in s
    ):
        return "stockout"
    if "already exists" in s or "resource already exists" in s:
        return "already_exists"
    if "permission" in s or "forbidden" in s or "not authorized" in s or "iam" in s:
        return "permission"
    if (
        "could not reach" in s
        or "connection" in s
        or "network is unreachable" in s
        or "timed out" in s
        or "timeout" in s
    ):
        return "network"
    return "other"


# --- project / billing / API setup checks -------------------------------------------

def project_state(project: str, *, timeout: float = 30) -> Optional[str]:
    """Return the project's lifecycle state (``ACTIVE`` etc.), or None if unreadable.

    None means the project could not be described at all — wrong ID, no access, or a
    gcloud/network failure. A reachable project with no state field reported comes back
    as an empty string (treated as "reachable, state unknown" by callers).
    """
    proc = _run(
        ["projects", "describe", project, "--format=value(lifecycleState)"],
        timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def billing_enabled(project: str, *, timeout: float = 30) -> tuple[Optional[bool], str]:
    """Return ``(enabled, detail)`` for the project's billing.

    ``enabled`` is True/False when we can tell, or None if the check itself failed
    (Billing API off, missing permission, network). ``detail`` carries the exact gcloud
    output/error so callers can show precisely what went wrong. A project with no billing
    account **cannot create any VM**, which is a common cause of "nothing ever provisions".
    """
    proc = _run(
        ["billing", "projects", "describe", project,
         "--format=value(billingEnabled)"],
        timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout).strip()
    val = proc.stdout.strip().lower()
    if val in ("true", "false"):
        return (val == "true"), proc.stdout.strip()
    return None, proc.stdout.strip()


def api_enabled(service: str, project: str, *, timeout: float = 30) -> tuple[Optional[bool], str]:
    """Return ``(enabled, detail)`` for whether ``service`` (e.g. ``compute.googleapis.com``)
    is enabled on the project. None if the check itself failed."""
    proc = _run(
        ["services", "list", "--enabled",
         f"--filter=config.name:{service}",
         "--format=value(config.name)",
         f"--project={project}"],
        timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout).strip()
    return (service in proc.stdout), proc.stdout.strip()


def enable_api(service: str, project: str, *, timeout: float = 300) -> tuple[bool, str]:
    """Enable ``service`` on the project. Returns ``(ok, detail)``; ``detail`` is the exact
    gcloud error when it fails.

    ``gcloud services enable`` is idempotent (enabling an already-enabled API succeeds), so
    this is safe to call whenever a check reports the API off. It can take ~30-60s to take
    effect. Failure is usually a missing ``serviceusage.services.enable`` permission or
    billing not being linked yet — both surfaced in ``detail``.
    """
    proc = _run(
        ["services", "enable", service, f"--project={project}"],
        timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    return True, proc.stdout.strip()


# --- quota / GPU health checks ------------------------------------------------------

def gpu_quota_metric(accelerator: str) -> str:
    """Map a GPU accelerator type to its Compute Engine quota metric name.

    e.g. ``nvidia-l4`` -> ``NVIDIA_L4_GPUS``. Unknown types fall back to the
    all-regions GPU metric so callers still get a usable filter.
    """
    a = accelerator.lower()
    if "l4" in a:
        return "NVIDIA_L4_GPUS"
    if "a100" in a:
        return "NVIDIA_A100_GPUS"
    if "t4" in a:
        return "NVIDIA_T4_GPUS"
    if "v100" in a:
        return "NVIDIA_V100_GPUS"
    return "GPUS_ALL_REGIONS"


def list_region_gpu_quota(
    metrics: Sequence[str],
    project: Optional[str] = None,
    *,
    timeout: float = 60,
) -> Optional[dict[str, dict[str, float]]]:
    """Return per-region available GPU quota for the requested quota metrics.

    Result is ``{region: {metric: available}}`` where ``available = limit - usage``
    (clamped at 0), including only regions/metrics whose limit is positive. This is a
    single ``regions list`` call, so it is cheap relative to probing every zone.

    Returns ``None`` if the quota query itself fails (gcloud missing, auth, API
    disabled, network) so callers can tell *"unknown"* apart from *"genuinely zero
    quota everywhere"* (an empty dict).
    """
    if not metrics:
        return {}
    metric_set = {m.upper() for m in metrics}
    filt = " OR ".join(f"quotas.metric={m}" for m in sorted(metric_set))
    args = [
        "compute", "regions", "list",
        "--flatten=quotas[]",
        f"--filter={filt}",
        "--format=csv[no-heading](name,quotas.metric,quotas.limit,quotas.usage)",
    ]
    if project:
        args.append(f"--project={project}")
    try:
        proc = _run(args, timeout=timeout, check=False)
    except (GcloudError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    result: dict[str, dict[str, float]] = {}
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        region, metric, limit_s, usage_s = parts[0], parts[1].upper(), parts[2], parts[3]
        if metric not in metric_set:
            continue
        try:
            limit = float(limit_s)
            usage = float(usage_s)
        except ValueError:
            continue
        if limit <= 0:
            continue
        avail = limit - usage
        result.setdefault(region, {})[metric] = avail if avail > 0 else 0.0
    return result

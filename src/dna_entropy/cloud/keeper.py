"""Always-on GPU keeper.

Secure a single GPU VM (``BOX_NAME``) in the user's own Google Cloud project and keep it
**RUNNING 24/7** until the user stops the keeper and deletes the VM by hand.

Design goals (per the professor's brief):
- **Never errors out.** Stockout, quota, auth, and transient gcloud failures are all
  treated as "try again later", not fatal. The loop runs forever with backoff.
- **Keeps the VM running**, not merely created. A watchdog re-checks the VM and restarts
  it if host maintenance (GPU VMs use ``--maintenance-policy=TERMINATE``) ever stops it.
- **Pre-installs the Evo stack** once the box is up, so the first real ``cloudrun`` is
  instant.

This module reuses the provisioning helpers in :mod:`orchestrator` (zone/offer escalation,
SSH wait, Evo install) so there is a single source of truth for how a box is built.

Run it standalone via ``keep_gpu.py`` at the repo root, or ``dna-entropy keep-gpu``.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import typer

from . import gcloud
from .orchestrator import (
    BOX_NAME,
    CloudConfig,
    _create_box,
    _ensure_evo,
    _quota_msg,
    _wait_for_ssh,
    load_state,
    preflight,
)

# Statuses that mean "the box is up (or coming up) — leave it alone".
_UP_STATES = frozenset({"RUNNING", "STAGING", "PROVISIONING", ""})
# Statuses that mean "the box exists but is not running — start it".
_STOPPED_STATES = frozenset({"TERMINATED", "STOPPED", "SUSPENDED"})

# Backoff bounds for the acquire retry loop (seconds).
_BACKOFF_START = 30
_BACKOFF_CAP = 300
# How often the watchdog re-checks a healthy box (seconds).
_POLL_DEFAULT = 60


def _next_action(existing: Optional[tuple[str, str]]) -> tuple[str, Optional[str]]:
    """Decide what to do given the current instance state.

    Args:
        existing: ``(zone, status)`` from :func:`gcloud.find_instance`, or ``None``.
    Returns:
        ``(action, zone)`` where action is ``"acquire"`` (no box -> create),
        ``"start"`` (box exists but stopped), or ``"up"`` (box is running/transitional).
    """
    if existing is None:
        return "acquire", None
    zone, status = existing
    if status in _STOPPED_STATES:
        return "start", zone
    # RUNNING / transitional / anything unknown -> assume it is (coming) up.
    return "up", zone


def _stamp() -> str:
    """Local timestamp for heartbeat lines (ASCII-only)."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _echo(msg: str, color: Optional[str] = None) -> None:
    typer.secho(f"  [{_stamp()}] {msg}", fg=color)


def _zone_region(zone: str) -> str:
    """``us-central1-a`` -> ``us-central1`` (quota is granted per region, not per zone)."""
    return zone.rsplit("-", 1)[0]


def _quota_metrics(cfg: CloudConfig) -> list[str]:
    """The distinct GPU quota metrics we care about, across all configured offers."""
    return sorted({gcloud.gpu_quota_metric(accel) for _machine, accel, _label in cfg.offers})


def _apply_quota_health(cfg: CloudConfig, all_zones: list[str], project: str) -> None:
    """Health-check GPU quota and narrow ``cfg.zones`` to regions that actually have it.

    The professor's symptom was a flood of "no quota in this region" / "no capacity"
    lines while the keeper probed ~90 zones blindly. Reading quota up front lets us:

    - **Only try zones in regions with quota** — the "no quota in this region" noise
      disappears, and any remaining failure is a genuine capacity stockout (transient).
    - **Say plainly when quota is zero everywhere** — the fixable case: it prints the
      one-time quota-request steps instead of looping silently.

    ``cfg.zones`` is reset from ``all_zones`` each call so a later quota grant is picked
    up automatically. If the quota query itself fails we can't tell, so we keep all zones.
    """
    quota = gcloud.list_region_gpu_quota(_quota_metrics(cfg), project)

    if quota is None:
        _echo("Could not read GPU quota; will try all regions.", color=typer.colors.YELLOW)
        cfg.zones = list(all_zones)
        return

    regions_with_quota = {r for r, metrics in quota.items() if any(v > 0 for v in metrics.values())}

    if not regions_with_quota:
        _echo("No GPU quota found in ANY region for this project (this is the fixable part).",
              color=typer.colors.RED)
        for line in _quota_msg("L4", "nvidia-l4").splitlines():
            _echo(line, color=typer.colors.CYAN)
        # Still try every zone in case the quota API under-reports; the create errors
        # will give the real per-zone reason.
        cfg.zones = list(all_zones)
        return

    eligible = [z for z in all_zones if _zone_region(z) in regions_with_quota]
    _echo(f"GPU quota available in {len(regions_with_quota)} region(s): "
          f"{', '.join(sorted(regions_with_quota))}", color=typer.colors.GREEN)
    # If none of our known zones fall in a quota region, fall back to all zones.
    cfg.zones = eligible or list(all_zones)


def _diagnose_setup(cfg: CloudConfig, account: str, project: str) -> bool:
    """Print a one-time GCP setup health report; return True if nothing blocking was found.

    This exists because a mis-set-up project (billing off, Compute Engine API off, wrong
    project, no quota) produces the exact symptom seen in the field: the console shows
    creates being attempted but nothing ever provisions. Each check prints ``OK:`` or a
    precise ``ERROR:`` with the exact gcloud text and the one command that fixes it, so a
    non-expert following the script can see *what* is wrong instead of a vague failure.

    Never raises — a diagnostic must not crash the keeper.
    """
    typer.secho("-" * 68, fg=typer.colors.CYAN)
    typer.secho("  GCP setup health check", fg=typer.colors.CYAN)
    typer.secho("-" * 68, fg=typer.colors.CYAN)
    ok = True

    # 1. Project reachable and ACTIVE.
    try:
        state = gcloud.project_state(project)
    except gcloud.GcloudError:
        state = None
    if state is None:
        _echo(f"ERROR: project '{project}' is not accessible from {account}.",
              color=typer.colors.RED)
        _echo("  Check the ID: gcloud config set project YOUR_PROJECT_ID", color=typer.colors.CYAN)
        ok = False
    elif state and state != "ACTIVE":
        _echo(f"ERROR: project '{project}' is {state}, not ACTIVE.", color=typer.colors.RED)
        ok = False
    else:
        _echo(f"OK: project '{project}' is reachable.", color=typer.colors.GREEN)

    # 2. Billing enabled (no billing -> zero VMs will ever create).
    try:
        billing, detail = gcloud.billing_enabled(project)
    except gcloud.GcloudError as exc:
        billing, detail = None, str(exc)
    if billing is True:
        _echo("OK: billing is enabled.", color=typer.colors.GREEN)
    elif billing is False:
        _echo("ERROR: billing is NOT enabled - GCP will not create any VM.", color=typer.colors.RED)
        _echo("  Fix: link a billing account at https://console.cloud.google.com/billing",
              color=typer.colors.CYAN)
        ok = False
    else:
        _echo(f"WARN: could not verify billing ({detail}).", color=typer.colors.YELLOW)
        _echo("  (Often just the Cloud Billing API being off; check the console if VMs never appear.)",
              color=typer.colors.CYAN)

    # 3. Compute Engine API enabled. If it's off, enable it automatically — nothing else
    #    (creating VMs, reading quota) works without it, and enabling is safe + idempotent.
    try:
        api, detail = gcloud.api_enabled("compute.googleapis.com", project)
    except gcloud.GcloudError as exc:
        api, detail = None, str(exc)
    if api is True:
        _echo("OK: Compute Engine API is enabled.", color=typer.colors.GREEN)
    elif api is False:
        _echo("Compute Engine API is not enabled - enabling it now (takes ~30-60s)...",
              color=typer.colors.YELLOW)
        try:
            enabled, err = gcloud.enable_api("compute.googleapis.com", project)
        except gcloud.GcloudError as exc:
            enabled, err = False, str(exc)
        if enabled:
            _echo("OK: enabled the Compute Engine API.", color=typer.colors.GREEN)
        else:
            _echo("ERROR: could not enable the Compute Engine API automatically.",
                  color=typer.colors.RED)
            _echo(f"  gcloud said: {err}", color=typer.colors.CYAN)
            _echo("  Fix it by hand: gcloud services enable compute.googleapis.com",
                  color=typer.colors.CYAN)
            ok = False
    else:
        _echo(f"WARN: could not verify the Compute Engine API ({detail}).", color=typer.colors.YELLOW)

    # 4. GPU quota (reuses the region quota health check).
    try:
        quota = gcloud.list_region_gpu_quota(_quota_metrics(cfg), project)
    except gcloud.GcloudError:
        quota = None
    if quota is None:
        _echo("WARN: could not read GPU quota.", color=typer.colors.YELLOW)
    elif not any(v > 0 for metrics in quota.values() for v in metrics.values()):
        _echo("ERROR: no GPU quota in any region (a one-time request is needed).",
              color=typer.colors.RED)
        for line in _quota_msg("L4", "nvidia-l4").splitlines():
            _echo(line, color=typer.colors.CYAN)
        ok = False
    else:
        regions = sorted(r for r, m in quota.items() if any(v > 0 for v in m.values()))
        _echo(f"OK: GPU quota present in {len(regions)} region(s): {', '.join(regions)}",
              color=typer.colors.GREEN)

    if ok:
        _echo("Setup looks good. Securing a GPU...", color=typer.colors.GREEN)
    else:
        _echo("Setup problem(s) found above. Fix them and leave the keeper running - "
              "it retries and will pick up automatically.", color=typer.colors.YELLOW)
    typer.secho("-" * 68, fg=typer.colors.CYAN)
    return ok


def _check_gpu(zone: str, cfg: CloudConfig, project: str) -> bool:
    """Confirm the box's GPU is actually alive (SSH-reachable != GPU-healthy).

    Runs ``nvidia-smi`` on the box. Returns True if a GPU is visible. Never raises —
    a health probe should not crash the keeper.
    """
    try:
        proc = gcloud.ssh(BOX_NAME, zone, "nvidia-smi -L", project=project,
                          key_file=cfg.ssh_key_file, timeout=60, check=False)
    except gcloud.GcloudError:
        return False
    return proc.returncode == 0 and "GPU" in (proc.stdout or "")


def _preflight_forever(cfg: CloudConfig, sleep: Callable[[float], None]) -> tuple[str, str]:
    """Run preflight, retrying forever on any failure (never raises).

    gcloud may be missing, unauthenticated, or the project unset. The user can fix any of
    these while the keeper is running, so we print the guidance and retry rather than exit.
    """
    while True:
        try:
            return preflight(cfg)
        except gcloud.GcloudError as exc:
            _echo(f"Not ready yet: {exc}", color=typer.colors.YELLOW)
            _echo("Fix the above, then leave this running - it will pick up automatically.",
                  color=typer.colors.CYAN)
            sleep(_BACKOFF_START)


def keep_alive(
    cfg: CloudConfig,
    *,
    install_evo: bool = True,
    poll: float = _POLL_DEFAULT,
    max_cycles: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Acquire a GPU box and keep it running forever.

    Args:
        cfg: cloud configuration (project, zones, offers, ssh key).
        install_evo: pre-install the Evo stack once the box is up (recommended).
        poll: seconds between watchdog checks of a healthy box.
        max_cycles: stop after this many loop iterations (tests only; ``None`` = forever).
        sleep: injectable sleep (tests pass a no-op / recorder).
    """
    account, project = _preflight_forever(cfg, sleep)
    if cfg.ssh_key_file is None:
        cfg.ssh_key_file = load_state().get("ssh_key_file")

    typer.secho("=" * 68, fg=typer.colors.CYAN)
    typer.secho("  DNA-Entropy GPU keeper - keeps a GPU VM running 24/7", fg=typer.colors.CYAN)
    typer.secho("=" * 68, fg=typer.colors.CYAN)
    _echo(f"account: {account}")
    _echo(f"project: {project}")
    _echo("Leave this window OPEN. The VM stays RUNNING (and billing) until you stop it.")
    _echo(f"To stop later: gcloud compute instances delete {BOX_NAME} --zone=<zone>",
          color=typer.colors.YELLOW)

    # One-time setup health check up front so a mis-configured project is diagnosed
    # immediately (billing/API/project/quota) instead of failing silently for hours.
    _diagnose_setup(cfg, account, project)

    backoff = _BACKOFF_START
    ready = False  # Evo stack confirmed installed on the current box
    cycles = 0
    all_zones = list(cfg.zones)  # full zone list; quota check narrows cfg.zones per attempt
    no_quota: set[tuple[str, str]] = set()
    first_run_done = False

    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        try:
            existing = gcloud.find_instance(BOX_NAME, project)
        except gcloud.GcloudError as exc:
            _echo(f"Could not query GCP ({exc}); retrying...", color=typer.colors.YELLOW)
            sleep_time = 10 if first_run_done else backoff
            sleep(sleep_time)
            if not first_run_done:
                backoff = min(backoff * 2, _BACKOFF_CAP)
            continue

        action, zone = _next_action(existing)

        if action == "acquire":
            # Health check first: only chase zones in regions that actually have quota,
            # so the log shows a real capacity picture instead of quota-denial spam.
            _apply_quota_health(cfg, all_zones, project)
            _echo("No GPU box found - trying to secure one where quota exists...")
            try:
                zone, _created = _create_box(project, load_state(), cfg, no_quota=no_quota)
                # Success! Reset search state
                backoff = _BACKOFF_START
                first_run_done = False
                no_quota.clear()
            except gcloud.GcloudError as exc:
                first_run_done = True
                # Print the exact gcloud text (setup remediation, per-kind errors, quota
                # steps) verbatim rather than a vague "no GPU" so the real cause is visible.
                for line in str(exc).splitlines():
                    _echo(line, color=typer.colors.YELLOW)
                _echo("Will keep trying...", color=typer.colors.CYAN)
                sleep(10)
                continue
            _echo(f"Secured a GPU box in {zone}.", color=typer.colors.GREEN)
            ready = False

        elif action == "start":
            _echo(f"Box in {zone} is stopped - starting it to keep it running...",
                  color=typer.colors.YELLOW)
            try:
                gcloud.start_vm(BOX_NAME, zone, project=project)
            except gcloud.GcloudError as exc:
                _echo(f"Start failed ({exc}); retrying...", color=typer.colors.YELLOW)
                sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_CAP)
                continue
            ready = False

        # The box should now be up. Ensure it is reachable and (once) has Evo installed.
        if not ready:
            try:
                _wait_for_ssh(BOX_NAME, zone, cfg, project)
                # SSH-reachable is not the same as GPU-healthy — verify the GPU is live.
                if not _check_gpu(zone, cfg, project):
                    _echo("Box is reachable but no GPU is visible (nvidia-smi found none); "
                          "will re-check.", color=typer.colors.YELLOW)
                    sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_CAP)
                    continue
                if install_evo:
                    _ensure_evo(BOX_NAME, zone, cfg, project)
                ready = True
                backoff = _BACKOFF_START
                first_run_done = False
                no_quota.clear()
                _echo(f"GPU is UP and healthy in {zone}. First run will be fast.",
                      color=typer.colors.GREEN)
            except gcloud.GcloudError as exc:
                _echo(f"Box not reachable yet ({exc}); will re-check.",
                      color=typer.colors.YELLOW)
                sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_CAP)
                continue

        _echo(f"heartbeat: GPU running in {zone}. (Ctrl-C to stop the keeper.)")
        sleep(poll)


def main(cfg: Optional[CloudConfig] = None) -> None:
    """Entry point for the standalone script and the ``keep-gpu`` CLI command."""
    try:
        keep_alive(cfg or CloudConfig())
    except KeyboardInterrupt:
        typer.secho(
            "\n  Keeper stopped. NOTE: the GPU VM is still RUNNING and billing.\n"
            f"  Delete it when done:  gcloud compute instances delete {BOX_NAME} --zone=<zone>",
            fg=typer.colors.YELLOW,
        )


if __name__ == "__main__":
    main()

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

    backoff = _BACKOFF_START
    ready = False  # Evo stack confirmed installed on the current box
    cycles = 0

    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        try:
            existing = gcloud.find_instance(BOX_NAME, project)
        except gcloud.GcloudError as exc:
            _echo(f"Could not query GCP ({exc}); retrying...", color=typer.colors.YELLOW)
            sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_CAP)
            continue

        action, zone = _next_action(existing)

        if action == "acquire":
            _echo("No GPU box found - trying to secure one across all zones...")
            try:
                zone, _created = _create_box(project, load_state(), cfg)
            except gcloud.GcloudError as exc:
                _echo(f"No GPU available right now ({exc}); will keep trying.",
                      color=typer.colors.YELLOW)
                sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_CAP)
                continue
            _echo(f"Secured a GPU box in {zone}.", color=typer.colors.GREEN)
            backoff = _BACKOFF_START
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
                if install_evo:
                    _ensure_evo(BOX_NAME, zone, cfg, project)
                ready = True
                backoff = _BACKOFF_START
                _echo(f"GPU is UP and ready in {zone}. First run will be fast.",
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

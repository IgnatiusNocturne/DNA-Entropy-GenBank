"""Tests for the cloud orchestrator's pure/local logic (no real GCP calls)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from dna_entropy.cloud import gcloud
from dna_entropy.cloud import orchestrator
from dna_entropy.cloud.orchestrator import (
    CloudConfig,
    _create_box,  # noqa: F401  (imported to ensure module loads)
    _ordered_zones,
    load_state,
    preflight,
    run_in_cloud,
    save_state,
)
from dna_entropy.cloud.ui import Spinner


# --- error classification -----------------------------------------------------------

@pytest.mark.parametrize(
    "stderr, expected",
    [
        ("Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0.0", "quota"),
        ("ZONE_RESOURCE_POOL_EXHAUSTED ... does not have enough resources", "stockout"),
        ("Required 'compute.instances.create' permission", "permission"),
        ("some unrelated failure", "other"),
        # Billing not enabled — a project-wide setup problem, must NOT read as capacity.
        ("The billing account for the owning project is disabled in state absent", "billing"),
        ("Billing must be enabled for activation of service(s)", "billing"),
        # Compute Engine API not enabled — likewise project-wide.
        ("Compute Engine API has not been used in project 123 before or it is disabled",
         "api_disabled"),
        ("accessNotConfigured: Compute Engine API is disabled", "api_disabled"),
        # Network / connectivity to gcloud.
        ("Could not reach the server; connection timed out", "network"),
    ],
)
def test_classify_create_error(stderr: str, expected: str) -> None:
    assert gcloud.classify_create_error(stderr) == expected


# --- quota / GPU health checks ------------------------------------------------------

@pytest.mark.parametrize(
    "accel, metric",
    [
        ("nvidia-l4", "NVIDIA_L4_GPUS"),
        ("nvidia-tesla-a100", "NVIDIA_A100_GPUS"),
        ("nvidia-tesla-t4", "NVIDIA_T4_GPUS"),
        ("something-weird", "GPUS_ALL_REGIONS"),
    ],
)
def test_gpu_quota_metric(accel: str, metric: str) -> None:
    assert gcloud.gpu_quota_metric(accel) == metric


def _fake_run(stdout: str, returncode: int = 0):
    import subprocess

    def _run(args, *, timeout=None, check=False, stdin_text=None):
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")
    return _run


def test_list_region_gpu_quota_parses_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # region,metric,limit,usage  (usage subtracted; limit<=0 dropped)
    out = "\n".join([
        "us-central1,NVIDIA_L4_GPUS,8.0,2.0",   # 6 available
        "us-east1,NVIDIA_L4_GPUS,0.0,0.0",       # no limit -> dropped
        "europe-west4,NVIDIA_A100_GPUS,4.0,4.0", # fully used -> 0
    ])
    monkeypatch.setattr(gcloud, "_run", _fake_run(out))
    quota = gcloud.list_region_gpu_quota(["NVIDIA_L4_GPUS", "NVIDIA_A100_GPUS"])
    assert quota == {
        "us-central1": {"NVIDIA_L4_GPUS": 6.0},
        "europe-west4": {"NVIDIA_A100_GPUS": 0.0},
    }


def test_list_region_gpu_quota_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-zero return means the query failed -> None ("unknown", not "zero everywhere").
    monkeypatch.setattr(gcloud, "_run", _fake_run("", returncode=1))
    assert gcloud.list_region_gpu_quota(["NVIDIA_L4_GPUS"]) is None


def test_list_region_gpu_quota_empty_metrics_is_empty() -> None:
    assert gcloud.list_region_gpu_quota([]) == {}


# --- project / billing / API setup checks -------------------------------------------

def test_project_state_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcloud, "_run", _fake_run("ACTIVE\n"))
    assert gcloud.project_state("proj") == "ACTIVE"


def test_project_state_none_when_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcloud, "_run", _fake_run("ERROR: not found", returncode=1))
    assert gcloud.project_state("proj") is None


@pytest.mark.parametrize(
    "stdout, rc, expected",
    [
        ("True\n", 0, True),
        ("False\n", 0, False),
        ("", 1, None),          # query failed -> unknown, not False
        ("garbage", 0, None),   # unexpected value -> unknown
    ],
)
def test_billing_enabled(monkeypatch: pytest.MonkeyPatch, stdout, rc, expected) -> None:
    monkeypatch.setattr(gcloud, "_run", _fake_run(stdout, returncode=rc))
    enabled, _detail = gcloud.billing_enabled("proj")
    assert enabled is expected


def test_api_enabled_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcloud, "_run", _fake_run("compute.googleapis.com\n"))
    enabled, _ = gcloud.api_enabled("compute.googleapis.com", "proj")
    assert enabled is True


def test_api_enabled_false_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcloud, "_run", _fake_run("\n"))  # empty -> not enabled
    enabled, _ = gcloud.api_enabled("compute.googleapis.com", "proj")
    assert enabled is False


def test_api_enabled_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcloud, "_run", _fake_run("permission denied", returncode=1))
    enabled, detail = gcloud.api_enabled("compute.googleapis.com", "proj")
    assert enabled is None and "permission denied" in detail


def test_enable_api_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcloud, "_run", _fake_run("Operation finished successfully.\n"))
    ok, _ = gcloud.enable_api("compute.googleapis.com", "proj")
    assert ok is True


def test_enable_api_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gcloud, "_run",
        _fake_run("PERMISSION_DENIED: serviceusage.services.enable", returncode=1),
    )
    ok, detail = gcloud.enable_api("compute.googleapis.com", "proj")
    assert ok is False and "serviceusage.services.enable" in detail


# --- _create_box surfaces setup problems ---------------------------------------------

def test_create_box_raises_billing_remediation_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """A billing error is project-wide: stop on the first zone, don't probe all ~90."""
    attempts = {"n": 0}

    def _attempt(zone, machine, accel, project, cfg):
        attempts["n"] += 1
        raise gcloud.GcloudError("Billing must be enabled for activation of service(s)")

    monkeypatch.setattr(orchestrator, "_attempt_create", _attempt)
    cfg = CloudConfig(zones=["us-central1-a", "us-central1-b", "us-east1-b", "us-west1-a"])

    with pytest.raises(gcloud.GcloudError) as ei:
        orchestrator._create_box("proj", {}, cfg)

    msg = str(ei.value)
    assert "Billing is not enabled" in msg      # remediation, not "no capacity"
    assert "gcloud said" in msg                 # exact gcloud text is included
    assert attempts["n"] == 1                    # bailed on the first zone


def test_create_box_final_message_includes_exact_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every zone is a genuine stockout, the give-up message shows the exact error."""
    def _attempt(zone, machine, accel, project, cfg):
        raise gcloud.GcloudError("ZONE_RESOURCE_POOL_EXHAUSTED: does not have enough resources")

    monkeypatch.setattr(orchestrator, "_attempt_create", _attempt)
    cfg = CloudConfig(zones=["z-a", "z-b"], offers=[("g2-standard-8", "nvidia-l4", "L4")])

    with pytest.raises(gcloud.GcloudError) as ei:
        orchestrator._create_box("proj", {}, cfg)

    msg = str(ei.value)
    assert "[stockout]" in msg
    assert "ZONE_RESOURCE_POOL_EXHAUSTED" in msg


def test_create_box_records_and_filters_quota_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted_zones = []

    def _attempt(zone, machine, accel, project, cfg):
        attempted_zones.append(zone)
        if zone == "us-central1-a":
            raise gcloud.GcloudError("Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0.0")
        else:
            raise gcloud.GcloudError("ZONE_RESOURCE_POOL_EXHAUSTED: does not have enough resources")

    monkeypatch.setattr(orchestrator, "_attempt_create", _attempt)
    cfg = CloudConfig(
        zones=["us-central1-a", "us-central1-b", "us-east1-a"],
        offers=[("g2-standard-8", "nvidia-l4", "L4")]
    )

    no_quota = set()
    with pytest.raises(gcloud.GcloudError):
        orchestrator._create_box("proj", {}, cfg, no_quota=no_quota)

    assert ("us-central1", "L4") in no_quota
    assert "us-central1-a" in attempted_zones
    assert "us-central1-b" in attempted_zones
    assert "us-east1-a" in attempted_zones

    attempted_zones.clear()

    with pytest.raises(gcloud.GcloudError):
        orchestrator._create_box("proj", {}, cfg, no_quota=no_quota)

    assert "us-east1-a" in attempted_zones
    assert "us-central1-a" not in attempted_zones
    assert "us-central1-b" not in attempted_zones


# --- spinner ------------------------------------------------------------------------

def test_spinner_non_tty_writes_start_and_end() -> None:
    buf = io.StringIO()  # StringIO.isatty() is False -> no thread, plain lines
    with Spinner("doing thing", stream=buf):
        pass
    out = buf.getvalue()
    assert "doing thing" in out
    assert "[OK]" in out


def test_spinner_marks_error_on_exception() -> None:
    buf = io.StringIO()
    with pytest.raises(ValueError):
        with Spinner("risky", stream=buf):
            raise ValueError("boom")
    assert "[!!]" in buf.getvalue()  # does not suppress, marks failure


# --- preflight ----------------------------------------------------------------------

def test_preflight_raises_when_gcloud_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcloud, "find_gcloud", lambda: None)
    with pytest.raises(gcloud.GcloudNotInstalled):
        preflight(CloudConfig())


def test_preflight_raises_when_not_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcloud, "find_gcloud", lambda: "gcloud")
    monkeypatch.setattr(gcloud, "active_account", lambda: None)
    with pytest.raises(gcloud.GcloudNotAuthenticated):
        preflight(CloudConfig())


# --- state + zone ordering ----------------------------------------------------------

def test_state_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    save_state({"last_zone": "us-east4-a"})
    assert load_state()["last_zone"] == "us-east4-a"


def test_load_state_missing_is_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "nope"))
    assert load_state() == {}


def test_ordered_zones_prefers_last_good() -> None:
    cfg = CloudConfig()
    ordered = _ordered_zones(cfg, {"last_zone": "us-east1-d"})
    assert ordered[0] == "us-east1-d"
    assert sorted(ordered) == sorted(cfg.zones)  # same set, just reordered


# --- run_in_cloud requires the keeper's box (never creates one) ----------------------

def test_run_in_cloud_errors_when_no_box(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(orchestrator, "preflight", lambda cfg: ("acct@example.com", "proj"))
    monkeypatch.setattr(orchestrator, "load_state", lambda: {})
    monkeypatch.setattr(gcloud, "find_instance", lambda *a, **k: None)
    # It must never try to create a VM.
    monkeypatch.setattr(orchestrator, "_create_box", lambda *a, **k: pytest.fail("must not create"))

    with pytest.raises(gcloud.GcloudError, match="keep_gpu"):
        run_in_cloud(seq="ACGT", name="x", base_dir=tmp_path, genes=False, cfg=CloudConfig())

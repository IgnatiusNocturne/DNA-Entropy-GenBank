"""Tests for the always-on GPU keeper's decision + loop logic (no real GCP calls).

We inject fakes for every gcloud/provisioning call and a no-op ``sleep``, and bound the
loop with ``max_cycles`` so the "forever" loop terminates in tests.
"""

from __future__ import annotations

import pytest

from dna_entropy.cloud import gcloud, keeper
from dna_entropy.cloud.keeper import (
    CloudConfig,
    _apply_quota_health,
    _check_gpu,
    _next_action,
    _quota_metrics,
    _zone_region,
    keep_alive,
)


# --- pure decision logic ------------------------------------------------------------

@pytest.mark.parametrize(
    "existing, expected_action, expected_zone",
    [
        (None, "acquire", None),
        (("us-central1-a", "RUNNING"), "up", "us-central1-a"),
        (("us-central1-a", ""), "up", "us-central1-a"),  # transitional / blank status
        (("us-central1-a", "STAGING"), "up", "us-central1-a"),
        (("us-central1-a", "TERMINATED"), "start", "us-central1-a"),
        (("us-central1-a", "STOPPED"), "start", "us-central1-a"),
        (("us-central1-a", "SUSPENDED"), "start", "us-central1-a"),
    ],
)
def test_next_action(existing, expected_action, expected_zone) -> None:
    action, zone = _next_action(existing)
    assert action == expected_action
    assert zone == expected_zone


# --- quota / GPU health checks ------------------------------------------------------

def test_zone_region() -> None:
    assert _zone_region("us-central1-a") == "us-central1"
    assert _zone_region("europe-west4-b") == "europe-west4"


def test_quota_metrics_covers_all_offers() -> None:
    # Default offers are L4 + A100.
    assert _quota_metrics(CloudConfig()) == ["NVIDIA_A100_GPUS", "NVIDIA_L4_GPUS"]


def test_apply_quota_health_narrows_to_quota_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = CloudConfig()
    all_zones = ["us-central1-a", "us-central1-b", "us-east1-b", "europe-west4-a"]
    monkeypatch.setattr(gcloud, "list_region_gpu_quota",
                        lambda *a, **k: {"us-central1": {"NVIDIA_L4_GPUS": 6.0}})
    _apply_quota_health(cfg, all_zones, "proj")
    assert cfg.zones == ["us-central1-a", "us-central1-b"]  # only the region with quota


def test_apply_quota_health_keeps_all_when_query_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = CloudConfig()
    all_zones = ["us-central1-a", "us-east1-b"]
    monkeypatch.setattr(gcloud, "list_region_gpu_quota", lambda *a, **k: None)
    _apply_quota_health(cfg, all_zones, "proj")
    assert cfg.zones == all_zones


def test_apply_quota_health_keeps_all_when_zero_quota_everywhere(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    cfg = CloudConfig()
    all_zones = ["us-central1-a", "us-east1-b"]
    monkeypatch.setattr(gcloud, "list_region_gpu_quota", lambda *a, **k: {})
    _apply_quota_health(cfg, all_zones, "proj")
    assert cfg.zones == all_zones  # still try; quota API can under-report
    assert "No GPU quota" in capsys.readouterr().out  # prints the fixable-case guidance


def test_check_gpu_true_when_nvidia_smi_lists_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def _ssh(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout="GPU 0: NVIDIA L4 (UUID: ...)", stderr="")

    monkeypatch.setattr(gcloud, "ssh", _ssh)
    assert _check_gpu("us-central1-a", CloudConfig(), "proj") is True


def test_check_gpu_false_when_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def _ssh(*a, **k):
        return subprocess.CompletedProcess(a, 1, stdout="", stderr="command not found")

    monkeypatch.setattr(gcloud, "ssh", _ssh)
    assert _check_gpu("us-central1-a", CloudConfig(), "proj") is False


# --- setup doctor -------------------------------------------------------------------

def _setup_env(monkeypatch, *, state="ACTIVE", billing=(True, "True"),
               api=(True, "compute.googleapis.com"), enable=(True, ""),
               quota=None):
    if quota is None:
        quota = {"us-central1": {"NVIDIA_L4_GPUS": 4.0}}
    monkeypatch.setattr(gcloud, "project_state", lambda *a, **k: state)
    monkeypatch.setattr(gcloud, "billing_enabled", lambda *a, **k: billing)
    monkeypatch.setattr(gcloud, "api_enabled", lambda *a, **k: api)
    monkeypatch.setattr(gcloud, "enable_api", lambda *a, **k: enable)
    monkeypatch.setattr(gcloud, "list_region_gpu_quota", lambda *a, **k: quota)


def test_diagnose_setup_all_ok(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _setup_env(monkeypatch)
    assert keeper._diagnose_setup(CloudConfig(), "a@b.com", "proj") is True
    out = capsys.readouterr().out
    assert "OK: billing is enabled" in out
    assert "OK: Compute Engine API is enabled" in out


def test_diagnose_setup_flags_billing_off(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _setup_env(monkeypatch, billing=(False, "False"))
    assert keeper._diagnose_setup(CloudConfig(), "a@b.com", "proj") is False
    assert "billing is NOT enabled" in capsys.readouterr().out


def test_diagnose_setup_auto_enables_api_when_off(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # API off but enabling succeeds -> not a blocker; the keeper self-heals.
    enable_calls = {"n": 0}

    def _enable(service, project, **k):
        enable_calls["n"] += 1
        assert service == "compute.googleapis.com"
        return True, ""

    _setup_env(monkeypatch, api=(False, ""))
    monkeypatch.setattr(gcloud, "enable_api", _enable)
    assert keeper._diagnose_setup(CloudConfig(), "a@b.com", "proj") is True
    assert enable_calls["n"] == 1
    assert "enabled the Compute Engine API" in capsys.readouterr().out


def test_diagnose_setup_reports_when_api_enable_fails(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # API off and enabling fails (e.g. no permission) -> blocker, exact error shown.
    _setup_env(monkeypatch, api=(False, ""),
               enable=(False, "PERMISSION_DENIED: serviceusage.services.enable"))
    assert keeper._diagnose_setup(CloudConfig(), "a@b.com", "proj") is False
    out = capsys.readouterr().out
    assert "could not enable the Compute Engine API" in out
    assert "serviceusage.services.enable" in out  # exact gcloud text surfaced


def test_diagnose_setup_flags_unreachable_project(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _setup_env(monkeypatch, state=None)
    assert keeper._diagnose_setup(CloudConfig(), "a@b.com", "proj") is False
    assert "not accessible" in capsys.readouterr().out


def test_diagnose_setup_flags_no_quota(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _setup_env(monkeypatch, quota={})
    assert keeper._diagnose_setup(CloudConfig(), "a@b.com", "proj") is False
    assert "no GPU quota in any region" in capsys.readouterr().out


def test_diagnose_setup_unknown_billing_is_warning_not_blocking(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # Billing API off -> we can't tell; that must WARN, not falsely block.
    _setup_env(monkeypatch, billing=(None, "Cloud Billing API has not been used"))
    assert keeper._diagnose_setup(CloudConfig(), "a@b.com", "proj") is True
    assert "could not verify billing" in capsys.readouterr().out


# --- loop behaviour -----------------------------------------------------------------

@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    """Neutralize all external calls; return a dict recording what the loop did."""
    calls = {"create": 0, "start": 0, "ssh": 0, "evo": 0, "sleeps": [], "stop": 0,
             "delete": 0, "gpu_check": 0}

    monkeypatch.setattr(keeper, "preflight", lambda cfg: ("acct@example.com", "proj"))
    monkeypatch.setattr(keeper, "load_state", lambda: {})

    def _wait_for_ssh(*a, **k):
        calls["ssh"] += 1

    def _ensure_evo(*a, **k):
        calls["evo"] += 1

    monkeypatch.setattr(keeper, "_wait_for_ssh", _wait_for_ssh)
    monkeypatch.setattr(keeper, "_ensure_evo", _ensure_evo)

    def _check_gpu(*a, **k):
        calls["gpu_check"] += 1
        return True

    monkeypatch.setattr(keeper, "_check_gpu", _check_gpu)
    # Neutralize the startup setup doctor + quota narrowing (no real gcloud in tests).
    monkeypatch.setattr(gcloud, "list_region_gpu_quota", lambda *a, **k: None)
    monkeypatch.setattr(gcloud, "project_state", lambda *a, **k: "ACTIVE")
    monkeypatch.setattr(gcloud, "billing_enabled", lambda *a, **k: (True, "True"))
    monkeypatch.setattr(gcloud, "api_enabled", lambda *a, **k: (True, "compute.googleapis.com"))
    monkeypatch.setattr(gcloud, "enable_api", lambda *a, **k: (True, ""))

    # The keeper must NEVER stop or delete the VM — blow up if it tries.
    def _forbidden(name):
        def _fn(*a, **k):
            raise AssertionError(f"keeper must not call gcloud.{name}")
        return _fn

    monkeypatch.setattr(gcloud, "stop_vm", _forbidden("stop_vm"))
    monkeypatch.setattr(gcloud, "delete_vm", _forbidden("delete_vm"))

    return calls


def _recording_sleep(calls):
    def _sleep(secs):
        calls["sleeps"].append(secs)
    return _sleep


def test_acquires_then_installs_once(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    calls = patched
    # No box on the first look; running thereafter.
    states = [None, ("us-central1-a", "RUNNING"), ("us-central1-a", "RUNNING")]
    monkeypatch.setattr(gcloud, "find_instance", lambda *a, **k: states.pop(0))

    def _create_box(project, state, cfg):
        calls["create"] += 1
        return "us-central1-a", True

    monkeypatch.setattr(keeper, "_create_box", _create_box)

    keep_alive(CloudConfig(), max_cycles=3, sleep=_recording_sleep(calls))

    assert calls["create"] == 1
    assert calls["evo"] == 1          # Evo installed exactly once for the box
    assert calls["ssh"] == 1
    assert calls["stop"] == 0 and calls["delete"] == 0


def test_retries_forever_when_no_gpu(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    calls = patched
    monkeypatch.setattr(gcloud, "find_instance", lambda *a, **k: None)

    def _create_box(project, state, cfg):
        calls["create"] += 1
        raise gcloud.GcloudError("no capacity in any zone")

    monkeypatch.setattr(keeper, "_create_box", _create_box)

    # Must not raise even though every acquire fails; it just keeps trying + backs off.
    keep_alive(CloudConfig(), max_cycles=3, sleep=_recording_sleep(calls))

    assert calls["create"] == 3
    assert len(calls["sleeps"]) == 3
    assert calls["sleeps"][0] < calls["sleeps"][-1]  # backoff grows


def test_starts_a_stopped_box(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    calls = patched
    states = [("us-central1-a", "TERMINATED"), ("us-central1-a", "RUNNING")]
    monkeypatch.setattr(gcloud, "find_instance", lambda *a, **k: states.pop(0))
    monkeypatch.setattr(keeper, "_create_box", lambda *a, **k: pytest.fail("should not create"))

    def _start_vm(name, zone, *, project=None, timeout=300):
        calls["start"] += 1

    monkeypatch.setattr(gcloud, "start_vm", _start_vm)

    keep_alive(CloudConfig(), max_cycles=2, sleep=_recording_sleep(calls))

    assert calls["start"] == 1
    assert calls["evo"] == 1


def test_rechecks_when_gpu_unhealthy(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    calls = patched
    monkeypatch.setattr(gcloud, "find_instance", lambda *a, **k: ("us-central1-a", "RUNNING"))
    monkeypatch.setattr(keeper, "_create_box", lambda *a, **k: ("us-central1-a", True))
    # GPU is dead this cycle -> keeper must NOT mark ready or install Evo; it backs off.
    monkeypatch.setattr(keeper, "_check_gpu", lambda *a, **k: False)

    keep_alive(CloudConfig(), max_cycles=1, sleep=_recording_sleep(calls))

    assert calls["evo"] == 0          # never installed onto a GPU-less box
    assert len(calls["sleeps"]) == 1  # backed off to re-check


def test_no_evo_install_when_disabled(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    calls = patched
    monkeypatch.setattr(gcloud, "find_instance", lambda *a, **k: ("us-central1-a", "RUNNING"))
    monkeypatch.setattr(keeper, "_create_box", lambda *a, **k: ("us-central1-a", True))

    keep_alive(CloudConfig(), install_evo=False, max_cycles=1, sleep=_recording_sleep(calls))

    assert calls["evo"] == 0
    assert calls["ssh"] == 1  # still confirms reachability

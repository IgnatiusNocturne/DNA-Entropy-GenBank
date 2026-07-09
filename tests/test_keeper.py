"""Tests for the always-on GPU keeper's decision + loop logic (no real GCP calls).

We inject fakes for every gcloud/provisioning call and a no-op ``sleep``, and bound the
loop with ``max_cycles`` so the "forever" loop terminates in tests.
"""

from __future__ import annotations

import pytest

from dna_entropy.cloud import gcloud, keeper
from dna_entropy.cloud.keeper import CloudConfig, _next_action, keep_alive


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


# --- loop behaviour -----------------------------------------------------------------

@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    """Neutralize all external calls; return a dict recording what the loop did."""
    calls = {"create": 0, "start": 0, "ssh": 0, "evo": 0, "sleeps": [], "stop": 0, "delete": 0}

    monkeypatch.setattr(keeper, "preflight", lambda cfg: ("acct@example.com", "proj"))
    monkeypatch.setattr(keeper, "load_state", lambda: {})

    def _wait_for_ssh(*a, **k):
        calls["ssh"] += 1

    def _ensure_evo(*a, **k):
        calls["evo"] += 1

    monkeypatch.setattr(keeper, "_wait_for_ssh", _wait_for_ssh)
    monkeypatch.setattr(keeper, "_ensure_evo", _ensure_evo)

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


def test_no_evo_install_when_disabled(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    calls = patched
    monkeypatch.setattr(gcloud, "find_instance", lambda *a, **k: ("us-central1-a", "RUNNING"))
    monkeypatch.setattr(keeper, "_create_box", lambda *a, **k: ("us-central1-a", True))

    keep_alive(CloudConfig(), install_evo=False, max_cycles=1, sleep=_recording_sleep(calls))

    assert calls["evo"] == 0
    assert calls["ssh"] == 1  # still confirms reachability

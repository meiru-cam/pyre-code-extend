"""The grader bounds how long a submission may run, and recovers cleanly when it does."""

from __future__ import annotations

import threading
import time

import pytest

import grading_service.main as gs


def _task(*cases: dict) -> dict:
    return {"function_name": "f", "tests": list(cases)}


def _case(name: str, code: str = "{fn}()\n", **extra) -> dict:
    return {"name": name, "code": code, **extra}


@pytest.fixture(autouse=True)
def short_deadline(monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(gs, "CASE_TIMEOUT_SECONDS", 0.5)
    return 0.5


HANGS = "def f():\n    while True:\n        pass\n"


@pytest.mark.parametrize("capture_output", [True, False])
def test_a_hanging_case_times_out_and_the_rest_are_not_run(capture_output):
    task = _task(_case("hangs"), _case("after"), _case("also after"))
    start = time.perf_counter()
    response = gs._execute_tests(HANGS, task, capture_output=capture_output)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"grading took {elapsed:.2f}s"
    first, *rest = response.results
    assert not first.passed and first.error.startswith("Timed out after 0.5s"), first.error
    assert first.errorScope == "solution", first
    assert first.errorLineText in {"while True:", "pass"}, first.errorLineText
    assert [r.error for r in rest] == [gs.NOT_RUN_AFTER_TIMEOUT] * 2
    assert response.passed == 0 and response.total == 3


def test_except_exception_in_the_submission_does_not_swallow_it():
    code = (
        "def f():\n"
        "    while True:\n"
        "        try:\n"
        "            sum(range(1000))\n"
        "        except Exception:\n"
        "            pass\n"
    )
    response = gs._execute_tests(code, _task(_case("retries forever")), capture_output=False)
    assert response.results[0].error.startswith("Timed out"), response.results[0].error


def test_one_except_baseexception_does_not_defeat_the_deadline():
    code = (
        "def f():\n"
        "    try:\n"
        "        while True:\n"
        "            pass\n"
        "    except BaseException:\n"
        "        pass\n"
        "    while True:\n"
        "        pass\n"
    )
    start = time.perf_counter()
    response = gs._execute_tests(code, _task(_case("swallows once")), capture_output=False)
    assert time.perf_counter() - start < 2.0
    assert response.results[0].error.startswith("Timed out"), response.results[0].error


def test_a_hanging_top_level_statement_is_reported_as_a_load_timeout():
    code = "def spin():\n    while True:\n        pass\n\nvalue = spin()\n\ndef f():\n    return 1\n"
    response = gs._execute_tests(code, _task(_case("never reached")))
    assert response.results == []
    assert response.error and response.error.startswith("Loading your code timed out"), response.error


def test_an_unshown_case_reports_the_timeout_not_its_failure_message():
    task = _task(
        _case("hidden", visibility="unshown", behavior="state.invariant",
              failure_message="The answer was wrong."),
        _case("hidden too", visibility="unshown", behavior="state.invariant",
              failure_message="Also wrong."),
    )
    response = gs._execute_tests(HANGS, task)
    assert response.results[0].error.startswith("Timed out"), response.results[0].error
    assert response.results[1].error == gs.NOT_RUN_AFTER_TIMEOUT


def test_a_timed_out_request_releases_the_lock_other_requests_need():
    passing = "def f():\n    return 1\n"
    finished = {}

    def hang():
        gs._execute_tests(HANGS, _task(_case("hangs")))

    def quick():
        start = time.perf_counter()
        response = gs._execute_tests(passing, _task(_case("quick")))
        finished["elapsed"] = time.perf_counter() - start
        finished["passed"] = response.allPassed

    hanging = threading.Thread(target=hang)
    hanging.start()
    time.sleep(0.1)  # let the hanging request take the stdout lock first
    waiting = threading.Thread(target=quick)
    waiting.start()
    hanging.join(3)
    waiting.join(3)
    assert not hanging.is_alive() and not waiting.is_alive()
    assert finished["passed"] is True
    assert finished["elapsed"] < 2.0, finished


def test_no_timeout_escapes_after_a_case_that_finishes_at_the_deadline(monkeypatch):
    # Cases that finish right around the deadline race the watchdog. Whatever it sent
    # must be caught inside the case or cleared, never delivered to grader code later.
    monkeypatch.setattr(gs, "CASE_TIMEOUT_SECONDS", 0.05)
    code = (
        "import time\n"
        "def f():\n"
        "    end = time.perf_counter() + 0.05\n"
        "    while time.perf_counter() < end:\n"
        "        pass\n"
    )
    for _ in range(20):
        gs._execute_tests(code, _task(_case("borderline")), capture_output=False)
        end = time.perf_counter() + gs._REINJECT_SECONDS * 1.5
        while time.perf_counter() < end:  # a stray GradingTimeout would surface here
            pass


def test_fast_submissions_are_unaffected():
    response = gs._execute_tests("def f():\n    return 2\n", _task(
        _case("one", "assert {fn}() == 2\n"),
        _case("two", "assert {fn}() + 1 == 3\n"),
    ))
    assert response.allPassed, response

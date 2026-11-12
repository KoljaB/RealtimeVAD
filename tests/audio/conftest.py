"""Runs every registered audio case once, in parallel, and prints what each test guards.

The audio is not in git. Get it with ``python tests/audio/fetch_audio.py`` and
``python tests/audio/fetch_backchannels.py``; without it these tests are skipped.
"""

import time

import pytest

import lab

GUARDS = []  # (test, what, measured, "<=" | ">=", limit, unit)
_timing = {}


class Guard:
    """Collects a test's numeric guards; ``done()`` fails with all that are broken."""

    def __init__(self, test):
        self.test, self.failed = test, []

    def __call__(self, what, measured, op, limit, unit=""):
        ok = measured <= limit if op == "<=" else measured >= limit
        GUARDS.append((self.test, what, measured, op, limit, unit, ok))
        if not ok:
            self.failed.append(f"{what}: {measured:g} {unit}, must be {op} {limit:g}")

    def done(self):
        assert not self.failed, "; ".join(self.failed)


@pytest.fixture
def guard(request):
    return Guard(request.node.name.removeprefix("test_"))


@pytest.fixture(scope="session")
def runs():
    """(references by clean file, results by case key) of every case in test_edge_cases.CASES."""
    if not lab.available():
        pytest.skip("test audio missing: run tests/audio/fetch_audio.py and tests/audio/fetch_backchannels.py")
    import test_edge_cases as suite

    t0 = time.perf_counter()
    cases = [dict(c) for c in suite.CASES.values()]
    speech = sorted({f for c in cases for f, _ in c["parts"] if "/speech/" in f.replace("\\", "/")
                     or "/whisper/" in f.replace("\\", "/")})
    refs, results = lab.run_all(cases, speech)
    _timing.update(cases=len(cases), audio=sum(r["seconds"] for r in results.values()), wall=time.perf_counter() - t0)
    return refs, results


def pytest_terminal_summary(terminalreporter):
    if not GUARDS:
        return
    tr = terminalreporter
    tr.section("audio edge cases: what is guarded")
    test = None
    for name, what, measured, op, limit, unit, ok in GUARDS:
        if name != test:
            test = name
            tr.write_line(name.replace("_", " "))
        tr.write_line(f"  {'ok  ' if ok else 'FAIL'} {what}: {measured:.3g} {unit} (must be {op} {limit:g})")
    if _timing:
        tr.write_line(f"{_timing['cases']} streams, {_timing['audio'] / 60:.0f} min of audio, detectors ran {_timing['wall']:.1f} s")

#!/usr/bin/env python3
"""
Minimal test runner for environments without pytest.

Implements just enough of the pytest surface used by this project:
fixtures (module scope), parametrize, approx, and skip. Real pytest is the
supported path -- ``pip install pytest && pytest -v`` -- this exists so the
suite can also run on a bare interpreter.

Usage:  python scripts/run_tests.py [-v]
"""

from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# pytest shim
# --------------------------------------------------------------------------

class _Skipped(Exception):
    pass


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, actual):
        tol = self.abs if self.abs is not None else (
            abs(self.expected) * self.rel if self.rel is not None else 1e-6
        )
        return abs(actual - self.expected) <= tol

    def __repr__(self):
        return f"approx({self.expected})"


class _Mark:
    @staticmethod
    def parametrize(argnames, argvalues):
        names = [a.strip() for a in argnames.split(",")]

        def deco(fn):
            fn._parametrize = (names, argvalues)
            return fn
        return deco

    def __getattr__(self, name):
        def deco(fn=None, **_kw):
            def tag(f):
                setattr(f, f"_{name}", True)
                return f
            return tag(fn) if fn is not None else tag
        return deco


class _PytestShim:
    approx = staticmethod(lambda e, rel=None, abs=None: _Approx(e, rel, abs))
    mark = _Mark()

    @staticmethod
    def fixture(fn=None, **kwargs):
        def wrap(f):
            f._is_fixture = True
            f._fixture_scope = kwargs.get("scope", "function")
            return f
        return wrap(fn) if fn is not None else wrap

    @staticmethod
    def skip(reason=""):
        raise _Skipped(reason)

    @staticmethod
    def raises(exc):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, et, ev, tb):
                if et is None:
                    raise AssertionError(f"expected {exc.__name__}")
                return issubclass(et, exc)
        return _Ctx()


if "pytest" not in sys.modules:
    try:
        import pytest  # noqa: F401
    except ImportError:
        sys.modules["pytest"] = _PytestShim()


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run_module(mod, verbose=False, skip_slow=False):
    fixtures, cache = {}, {}
    tests = []

    for name, obj in vars(mod).items():
        if not callable(obj):
            continue
        if getattr(obj, "_is_fixture", False):
            fixtures[name] = obj
        elif name.startswith("test_"):
            tests.append((name, obj))

    def resolve(fname):
        if fname in cache:
            return cache[fname]
        fn = fixtures[fname]
        args = [resolve(p) for p in inspect.signature(fn).parameters]
        val = fn(*args)
        if getattr(fn, "_fixture_scope", "function") == "module":
            cache[fname] = val
        return val

    passed = failed = skipped = 0
    failures = []

    for name, fn in tests:
        if skip_slow and getattr(fn, "_slow", False):
            skipped += 1
            continue
        param = getattr(fn, "_parametrize", None)
        cases = []
        if param:
            names, values = param
            for v in values:
                v = v if isinstance(v, (tuple, list)) else (v,)
                cases.append((dict(zip(names, v)), f"{name}[{'-'.join(map(str, v))}]"))
        else:
            cases.append(({}, name))

        for kwargs, label in cases:
            try:
                sig = inspect.signature(fn)
                call_kwargs = dict(kwargs)
                for p in sig.parameters:
                    if p not in call_kwargs and p in fixtures:
                        call_kwargs[p] = resolve(p)
                fn(**call_kwargs)
                passed += 1
                if verbose:
                    print(f"  PASS  {label}")
            except _Skipped as e:
                skipped += 1
                if verbose:
                    print(f"  SKIP  {label}: {e}")
            except Exception:
                failed += 1
                failures.append((label, traceback.format_exc()))
                print(f"  FAIL  {label}")

    return passed, failed, skipped, failures


def main():
    verbose = "-v" in sys.argv
    skip_slow = "--fast" in sys.argv
    import importlib

    total_p = total_f = total_s = 0
    all_failures = []

    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    if not test_files:
        print("no test files found")
        return 1

    for path in test_files:
        print(f"\n{path.name}")
        mod = importlib.import_module(f"tests.{path.stem}")
        p, f, s, fails = run_module(mod, verbose, skip_slow)
        total_p += p
        total_f += f
        total_s += s
        all_failures.extend(fails)
        print(f"  -> {p} passed, {f} failed, {s} skipped")

    if all_failures:
        print("\n" + "=" * 70)
        for label, tb in all_failures:
            print(f"\nFAILURE: {label}\n{tb}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_p} passed, {total_f} failed, {total_s} skipped")
    return 1 if total_f else 0


if __name__ == "__main__":
    sys.exit(main())

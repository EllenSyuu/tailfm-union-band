"""The appendix identities (scores, efficiency, bias factor) hold symbolically."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sympy")


def test_verify_lemmas_symbolic():
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_lemmas_symbolic.py"
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr

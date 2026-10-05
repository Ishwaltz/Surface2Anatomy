"""Tests CLI commands."""

import subprocess
import sys

def test_cli_help():
    cmd = [sys.executable, "-m", "surface2anatomy.cli", "--help"]
    out = subprocess.check_output(cmd, text=True)
    assert "surface2anatomy" in out
    assert "predict" in out

def test_cli_info():
    cmd = [sys.executable, "-m", "surface2anatomy.cli", "info"]
    out = subprocess.check_output(cmd, text=True)
    assert "Surface2Anatomy" in out
    assert "Target-conditioned" in out

def test_cli_list_targets():
    cmd = [sys.executable, "-m", "surface2anatomy.cli", "list-targets", "--category", "Head & Brain"]
    out = subprocess.check_output(cmd, text=True)
    assert "brain" in out
    assert "skull" in out

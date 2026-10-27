"""hamsterball-bench (spec §14): env steps/s per configuration and the M2 gate line."""

import re

import pytest

from click.testing import CliRunner

from hamsterball.bench import main


@pytest.mark.gl  # the pixels config renders
def test_the_bench_prints_every_configuration_and_the_gate():
    result = CliRunner().invoke(main, ["--secs", "0.3", "--reps", "1", "--copies", "2"])
    rows = re.findall(r"^(state|pixels)\s+(flat|hfield)\s+(\d+)\s+[\d,]+$", result.output, re.M)
    assert {(c, lv, int(n)) for c, lv, n in rows} == {
        (c, lv, n) for c in ("state", "pixels") for lv in ("flat", "hfield") for n in (1, 2)
    }
    gate = re.search(r"^gate: .*: [\d,]+ env steps/s, (PASS|FAIL)$", result.output, re.M)
    assert gate, result.output
    assert result.exit_code == (0 if gate[1] == "PASS" else 1)


@pytest.mark.gl
def test_the_bench_runs_the_articulated_tier_and_prints_throughput_with_no_gate():
    result = CliRunner().invoke(main, ["--tier", "articulated", "--secs", "0.3", "--reps", "1", "--copies", "2"])
    assert result.exit_code == 0, result.output
    rows = re.findall(r"^(state|pixels)\s+(flat|hfield)\s+(\d+)\s+[\d,]+$", result.output, re.M)
    assert len(rows) == 8
    assert "articulated" in result.output and "gate" not in result.output

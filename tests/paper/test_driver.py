"""The driver: fixed build order, every step importable with a main(argv), command-line contract."""
import importlib

import pytest

from paper import make_paper


def test_steps_are_importable_and_have_main():
    for mod, what in make_paper.STEPS:
        m = importlib.import_module(f"paper.{mod}")
        assert callable(getattr(m, "main", None)), mod
        assert what


def test_build_order_constraints():
    order = [m for m, _ in make_paper.STEPS]
    assert len(order) == len(set(order))
    assert order[0] == "closed_build" and order[-1] == "stats.make_numbers"
    before = lambda a, b: order.index(a) < order.index(b)        # noqa: E731
    assert before("stats.openset_operating_point", "figures.open_heatmap")          # the heatmap reads the operating-point table
    assert before("closed_build", "figures.showcase") and before("closed_build", "figures.closed_pr")         # both read closed/numbers.json
    assert before("figures.zero_one_vs_pairs", "stats.make_numbers")


def test_list_and_unknown_step(capsys):
    assert make_paper.main(["--list"]) == 0
    assert "closed_build" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        make_paper.main(["--only", "figures.nonexistent"])


def test_run_stops_at_the_first_failing_step(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, name, fail):
            self.name, self.fail = name, fail

        def main(self, argv=None):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("boom")

    fakes = {f"paper.{m}": Fake(m, m == "figures.corpus") for m, _ in make_paper.STEPS}
    monkeypatch.setattr(make_paper.importlib, "import_module", lambda n: fakes[n])
    res = make_paper.run(None, verbose=False)
    assert res[-1][0] == "figures.corpus" and res[-1][2] is not None
    assert "stats.make_numbers" not in calls and calls[0] == "closed_build"

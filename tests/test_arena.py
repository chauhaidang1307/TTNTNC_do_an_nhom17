import csv
import json

import pytest

from arena import tournament
from arena.match import play_game
from arena.metrics import percentile, score, wilson
from agents.random_agent import RandomAgent
from engine.board import PLAYER_1, PLAYER_2


# ---------- metrics ----------

def test_score_formula():
    assert score(3, 2, 10) == 0.4  # (3 + 0,5*2) / 10
    assert score(0, 0, 0) == 0.0


def test_wilson_basic_properties():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(50, 100)
    assert 0 <= lo < 0.5 < hi <= 1
    assert abs((lo + hi) / 2 - 0.5) < 1e-9  # đối xứng quanh 0,5
    lo_all, hi_all = wilson(100, 100)
    assert hi_all == pytest.approx(1.0) and 0.9 < lo_all < 1.0  # không vượt 1 và không suy biến


def test_wilson_narrows_with_more_games():
    w_small = wilson(5, 10)
    w_big = wilson(500, 1000)
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([7], 0.95) == 7


# ---------- match ----------

def test_play_game_returns_consistent_record():
    winner, moves = play_game(RandomAgent(seed=1), RandomAgent(seed=2))
    assert winner in (PLAYER_1, PLAYER_2, None)
    assert [m["ply"] for m in moves] == list(range(1, len(moves) + 1))
    assert all(m["player"] == (PLAYER_1 if m["ply"] % 2 == 1 else PLAYER_2) for m in moves)
    assert all(0 <= m["col"] < 7 and m["time_ms"] >= 0 for m in moves)
    assert all(m["nodes"] == "" and m["depth"] == "" for m in moves)  # Random không có chỉ số chi phí


def test_play_game_records_cost_of_alphabeta():
    a = tournament.make_agent("ab-d2", 0)
    _, moves = play_game(a, RandomAgent(seed=0))
    ab_moves = [m for m in moves if m["agent"] == a.name]
    assert ab_moves and all(m["nodes"] > 0 and m["depth"] == 2 for m in ab_moves)


# ---------- make_agent ----------

@pytest.mark.parametrize("spec, name", [
    ("random", "Random"), ("heuristic", "Heuristic"),
    ("ab-d3", "AB-d3"), ("ab-d3-nopr", "AB-d3-nopr"),
    ("ab-t50", "AB-t50ms"), ("ab-t50-nopr", "AB-t50ms-nopr"),
])
def test_make_agent_names(spec, name):
    assert tournament.make_agent(spec, 0).name == name


def test_make_agent_rejects_unknown():
    with pytest.raises(ValueError):
        tournament.make_agent("mcts-xx", 0)


def test_make_agent_time_mode_settings():
    a = tournament.make_agent("ab-t100", 0)
    assert a.time_limit == pytest.approx(0.1) and a.depth == 42


# ---------- run_pair ----------

def run(spec_a, spec_b, games, split="dev"):
    g, m = [], []
    tournament.run_pair("EXP-T", split, spec_a, spec_b, games, g, m)
    return g, m


def test_run_pair_balances_first_and_second():
    g, _ = run("random", "heuristic", 20)
    assert sum(r["a_first"] for r in g) == 10
    assert len(g) == 20


def test_run_pair_rejects_odd_games():
    with pytest.raises(ValueError):
        run("random", "heuristic", 5)


def test_run_pair_winner_labels_match_moves():
    g, m = run("random", "heuristic", 10)
    for r in g:
        assert r["winner"] in ("random", "heuristic", "draw")
        assert r["moves"] == sum(1 for x in m if x["game"] == r["game"])


def test_run_pair_reproducible_in_fixed_depth_mode():
    g1, _ = run("ab-d2", "random", 6)
    g2, _ = run("ab-d2", "random", 6)
    assert [(r["winner"], r["moves"]) for r in g1] == [(r["winner"], r["moves"]) for r in g2]


def test_dev_and_test_seed_ranges_do_not_overlap():
    dev, _ = run("random", "heuristic", 10, split="dev")
    test, _ = run("random", "heuristic", 10, split="test")
    dev_seeds = {r["seed_a"] for r in dev} | {r["seed_b"] for r in dev}
    test_seeds = {r["seed_a"] for r in test} | {r["seed_b"] for r in test}
    assert not (dev_seeds & test_seeds)
    assert not ({r["seed_a"] for r in dev} & {r["seed_b"] for r in dev})  # hai bên không dùng chung seed


def test_summary_counts_add_up():
    g, _ = run("heuristic", "random", 10)
    s = tournament.summarize(g, "heuristic")
    t = s["total"]
    assert t["win"] + t["draw"] + t["loss"] == t["games"] == 10
    assert s["a_first"]["games"] == s["a_second"]["games"] == 5
    assert 0 <= t["wilson_low"] <= t["score"] <= t["wilson_high"] <= 1


# ---------- main (ghi log vào thư mục tạm) ----------

def run_main(monkeypatch, tmp_path, *argv):
    monkeypatch.setattr(tournament, "LOGS", tmp_path)
    monkeypatch.setattr("sys.argv", ["tournament", *argv])
    tournament.main()


def test_main_writes_separate_files_per_pair(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path, "--exp", "EXP-T", "--games", "4", "--agents", "random", "heuristic", "ab-d2")
    pairs = [("random", "heuristic"), ("random", "ab-d2"), ("heuristic", "ab-d2")]
    assert [d.name for d in tmp_path.iterdir()] == ["EXP-T"]  # mọi thứ nằm trong thư mục của thí nghiệm
    assert sorted(d.name for d in (tmp_path / "EXP-T").iterdir()) == sorted(f"{a}_vs_{b}" for a, b in pairs)
    for a, b in pairs:
        assert len(list((tmp_path / "EXP-T" / f"{a}_vs_{b}").iterdir())) == 3  # mỗi cặp đúng ba file
        for kind in ("games.csv", "moves.csv", "meta.json"):
            assert (tmp_path / "EXP-T" / f"{a}_vs_{b}" / f"EXP-T_dev_{a}_vs_{b}_{kind}").exists()
        with open(tmp_path / "EXP-T" / f"{a}_vs_{b}" / f"EXP-T_dev_{a}_vs_{b}_games.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert {r["pair"] for r in rows} == {f"{a}__vs__{b}"}  # chỉ chứa đúng một cặp
        meta = json.loads((tmp_path / "EXP-T" / f"{a}_vs_{b}" / f"EXP-T_dev_{a}_vs_{b}_meta.json").read_text(encoding="utf-8"))
        assert meta["agent_a"] == a and meta["agent_b"] == b and meta["games"] == 4
        t = meta["summary"]["total"]
        assert t["win"] + t["draw"] + t["loss"] == 4


def test_main_overwrite_check_happens_before_any_pair_runs(monkeypatch, tmp_path):
    # Đã có file của cặp thứ hai: phải từ chối ngay, không ghi cặp thứ nhất.
    folder = tmp_path / "EXP-T" / "random_vs_ab-d2"
    folder.mkdir(parents=True)
    (folder / "EXP-T_dev_random_vs_ab-d2_games.csv").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--exp", "EXP-T", "--games", "2", "--agents", "random", "heuristic", "ab-d2")
    assert [p.name for p in folder.iterdir()] == ["EXP-T_dev_random_vs_ab-d2_games.csv"]
    assert [d.name for d in (tmp_path / "EXP-T").iterdir()] == ["random_vs_ab-d2"]  # không tạo thư mục cặp khác


def test_main_refuses_to_overwrite(monkeypatch, tmp_path):
    args = ("--exp", "EXP-T", "--games", "2", "--agents", "random", "heuristic")
    run_main(monkeypatch, tmp_path, *args)
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, *args)


def test_main_test_split_needs_confirmation(monkeypatch, tmp_path):
    args = ("--exp", "EXP-T", "--split", "test", "--games", "2", "--agents", "random", "heuristic")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, *args)
    assert not list(tmp_path.iterdir())
    run_main(monkeypatch, tmp_path, *args, "--confirm-test")
    assert (tmp_path / "EXP-T" / "random_vs_heuristic" / "EXP-T_test_random_vs_heuristic_games.csv").exists()


def test_main_rejects_bad_agent_before_running(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        run_main(monkeypatch, tmp_path, "--exp", "EXP-T", "--games", "2", "--agents", "random", "nope")
    assert not list(tmp_path.iterdir())


def test_winner_label_agrees_with_replayed_moves():
    # Phát lại từng ván từ log nước đi trên một Board mới, đối chiếu người thắng thật với nhãn trong log ván.
    from engine.board import Board
    g, m = run("ab-d2", "random", 8)
    ab_name = tournament.make_agent("ab-d2", 0).name
    for r in g:
        b = Board()
        who = {}
        for x in (x for x in m if x["game"] == r["game"]):
            who[x["player"]] = x["agent"]
            b.step(x["col"])
        assert b.is_terminal()
        if b.winner is None:
            assert r["winner"] == "draw"
        else:
            expected = "ab-d2" if who[b.winner] == ab_name else "random"
            assert r["winner"] == expected
        assert r["a_first"] == int(who[PLAYER_1] == ab_name)


def test_wilson_stays_within_unit_interval():
    for n in (1, 2, 10, 100):
        for k in (0, n):
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= hi <= 1.0

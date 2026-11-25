import copy
import csv
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from agents.dqn_agent import DQNAgent, greedy_action
from arena import tournament
from arena.match import play_game
from agents.random_agent import RandomAgent
from engine import board
from engine.board import COLS, ROWS
from engine.encoding import encode, legal_mask
from rl import trainer
from rl.checkpoint import load_network, save_checkpoint
from rl.network import QNet
from rl.replay import ReplayBuffer

P1, P2 = board.PLAYER_1, board.PLAYER_2


def play(moves):
    b = board.Board()
    for col in moves:
        b.step(col)
    return b


class FixedQ(nn.Module):
    # Mạng giả: luôn trả về cùng một bảng Q (để kiểm tra phép tính mục tiêu bằng tay).
    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table, dtype=torch.float32)

    def forward(self, x):
        return self.table.unsqueeze(0).repeat(x.shape[0], 1)


# ---------- mã hóa ----------

def test_encode_shape_and_values():
    x = encode(play([3, 4]))  # người 1 ở (0,3), người 2 ở (0,4), đến lượt người 1
    assert x.shape == (2, ROWS, COLS) and x.dtype == np.uint8
    assert x[0, 0, 3] == 1 and x[1, 0, 4] == 1
    assert x.sum() == 2


def test_encode_is_from_perspective_of_player_to_move():
    b = play([3])  # người 2 đến lượt; quân của người 1 là "đối thủ" của người đến lượt
    x = encode(b)
    assert x[1, 0, 3] == 1 and x[0].sum() == 0


def test_encode_swaps_channels_when_perspective_changes():
    b = play([3, 4])
    x1 = encode(b)
    b.step(0)
    x2 = encode(b)
    assert x2[1, 0, 3] == 1 and x2[0, 0, 4] == 1  # sau một nước, vai trò hai kênh đảo cho các quân cũ
    assert x1[0, 0, 3] == 1 and x1[1, 0, 4] == 1


def test_encode_does_not_change_board():
    b = play([3, 3, 2])
    before = [row[:] for row in b.grid]
    encode(b)
    legal_mask(b)
    assert b.grid == before


def test_legal_mask_excludes_full_columns_and_terminal():
    b = play([3] * 6)
    m = legal_mask(b)
    assert m.dtype == bool and not m[3] and m.sum() == COLS - 1
    won = play([0, 1, 0, 1, 0, 1, 0])
    assert not legal_mask(won).any()


# ---------- mạng ----------

def test_network_output_shape():
    net = QNet(channels=8, hidden=16)
    out = net(torch.zeros(5, 2, ROWS, COLS))
    assert out.shape == (5, COLS)


# ---------- replay ----------

def test_replay_wraps_around_capacity():
    rb = ReplayBuffer(4, seed=0)
    for i in range(10):
        rb.add(np.zeros((2, ROWS, COLS), np.uint8), i % COLS, float(i), np.zeros((2, ROWS, COLS), np.uint8),
               False, np.ones(COLS, bool))
    assert len(rb) == 4
    assert sorted(rb.rewards[:4]) == [6.0, 7.0, 8.0, 9.0]  # chỉ còn 4 bước chuyển mới nhất


def test_replay_mirror_flips_state_action_and_mask_consistently():
    rb = ReplayBuffer(8, seed=0)
    s = np.zeros((2, ROWS, COLS), np.uint8)
    s[0, 0, 1] = 1  # quân ở cột 1
    ns = np.zeros((2, ROWS, COLS), np.uint8)
    ns[1, 0, 2] = 1
    mask = np.array([1, 1, 0, 1, 1, 1, 1], bool)  # cột 2 đầy
    rb.add(s, 1, 0.0, ns, False, mask)
    for flip, (col_s, act, col_ns, full_col) in {False: (1, 1, 2, 2), True: (5, 5, 4, 4)}.items():
        bs, ba, _, bns, _, bnm = rb.sample(1, "cpu", flip=[flip])
        assert bs[0, 0, 0, col_s] == 1 and ba[0] == act and bns[0, 1, 0, col_ns] == 1
        assert not bnm[0, full_col] and bnm[0].sum() == COLS - 1


# ---------- phần thưởng và bước chuyển do play_episode ghi ----------

def scripted(moves):
    it = iter(moves)
    return lambda b: next(it)


def test_play_episode_records_every_move_with_correct_reward():
    rb = ReplayBuffer(100, seed=0)
    # người 1 thả cột 0 bốn lần (thắng ở nước thứ 7), người 2 thả cột 1
    policies = {P1: scripted([0, 0, 0, 0]), P2: scripted([1, 1, 1])}
    winner, plies = trainer.play_episode(policies, rb)
    assert winner == P1 and plies == 7 and len(rb) == 7
    assert list(rb.rewards[:7]) == [0, 0, 0, 0, 0, 0, 1.0]  # chỉ nước thắng có thưởng +1
    assert list(rb.dones[:7]) == [False] * 6 + [True]
    assert rb.actions[0] == 0 and rb.actions[1] == 1
    assert not rb.next_masks[6].any()  # sau nước thắng, không còn nước hợp lệ
    assert rb.states[0].sum() == 0  # trạng thái đầu là bàn trống


def test_play_episode_draw_has_zero_reward():
    from tests.test_mcts import DRAW_GAME
    rb = ReplayBuffer(100, seed=0)
    winner, plies = trainer.play_episode({P1: scripted(DRAW_GAME[0::2]), P2: scripted(DRAW_GAME[1::2])}, rb)
    assert winner is None and plies == 42
    assert rb.rewards[:42].sum() == 0 and rb.dones[41] and not rb.dones[:41].any()


# ---------- mục tiêu Double DQN kiểu negamax (dấu!) ----------

def batch(reward, done, next_mask, n=1):
    r = torch.full((n,), reward)
    d = torch.full((n,), float(done))
    ns = torch.zeros(n, 2, ROWS, COLS)
    nm = torch.tensor([next_mask] * n, dtype=torch.bool)
    return r, d, ns, nm


def test_target_is_negated_next_value():
    online = FixedQ([0, 0, 0, 0.9, 0, 0, 0])
    target = FixedQ([0, 0, 0, 0.5, 0, 0, 0])
    r, d, ns, nm = batch(0.0, False, [True] * 7)
    y = trainer.compute_targets(online, target, r, d, ns, nm, gamma=0.99)
    assert y.item() == pytest.approx(-0.99 * 0.5)  # đổi dấu vì s' là góc nhìn của đối thủ


def test_target_for_terminal_is_just_reward():
    online = FixedQ([9] * 7)
    target = FixedQ([9] * 7)
    r, d, ns, nm = batch(1.0, True, [False] * 7)
    y = trainer.compute_targets(online, target, r, d, ns, nm, gamma=0.99)
    assert y.item() == pytest.approx(1.0)


def test_double_dqn_selects_with_online_and_evaluates_with_target():
    online = FixedQ([0, 0, 5.0, 0, 1.0, 0, 0])   # online thích cột 2
    target = FixedQ([0, 0, 0.2, 0, 8.0, 0, 0])   # target thích cột 4 (không được dùng để chọn)
    r, d, ns, nm = batch(0.0, False, [True] * 7)
    y = trainer.compute_targets(online, target, r, d, ns, nm, gamma=1.0)
    assert y.item() == pytest.approx(-0.2)  # = -Q_target(s', argmax Q_online) = -Q_target[2]


def test_target_ignores_illegal_next_actions():
    online = FixedQ([0, 0, 5.0, 0, 1.0, 0, 0])
    target = FixedQ([0, 0, 0.2, 0, 3.0, 0, 0])
    r, d, ns, nm = batch(0.0, False, [True, True, False, True, True, True, True])  # cột 2 đầy
    y = trainer.compute_targets(online, target, r, d, ns, nm, gamma=1.0)
    assert y.item() == pytest.approx(-3.0)  # cột 2 bị loại nên online chọn cột 4


# ---------- tác tử DQN ----------

def test_greedy_never_picks_full_column():
    b = play([3] * 6)
    net = FixedQ([0, 0, 0, 100.0, 0, 0, 0])  # mạng "muốn" cột đã đầy
    assert greedy_action(net, b) != 3
    assert DQNAgent(net).choose_action(b) in b.legal_actions()


def test_dqn_agent_raises_on_finished_game():
    with pytest.raises(ValueError):
        DQNAgent(QNet(4, 8)).choose_action(play([0, 1, 0, 1, 0, 1, 0]))


def test_dqn_agent_plays_legal_full_games():
    net = QNet(4, 8)
    for i in range(3):
        winner, moves = play_game(DQNAgent(net), RandomAgent(seed=i))
        assert all(0 <= m["col"] < COLS for m in moves)


def test_epsilon_one_is_random_but_legal():
    b = play([3] * 6)
    ag = DQNAgent(FixedQ([0, 0, 0, 100.0, 0, 0, 0]), epsilon=1.0, seed=0)
    assert all(ag.choose_action(b) != 3 for _ in range(30))


# ---------- checkpoint ----------

def test_checkpoint_roundtrip_gives_same_outputs(tmp_path):
    net = QNet(channels=8, hidden=16)
    path = tmp_path / "x.pt"
    save_checkpoint(path, net, {"channels": 8, "hidden": 16}, episode=7, extra={"k": 1})
    loaded = load_network(str(path), "cpu")
    x = torch.randn(3, 2, ROWS, COLS)
    net.eval()
    assert torch.allclose(net(x), loaded(x))
    assert torch.load(path)["episode"] == 7


# ---------- lịch ----------

CFG = trainer.load_config("configs/rl_v1.json")


def test_stage_schedule_matches_config():
    assert trainer.total_episodes(CFG) == 200000
    assert trainer.stage_at(0, CFG["stages"])["name"] == "vs_random"
    assert trainer.stage_at(19999, CFG["stages"])["name"] == "vs_random"
    assert trainer.stage_at(20000, CFG["stages"])["name"] == "vs_heuristic"
    assert trainer.stage_at(60000, CFG["stages"])["name"] == "self_pool"
    assert trainer.stage_at(10**9, CFG["stages"])["name"] == "self_pool"


def test_epsilon_and_lr_schedules():
    assert trainer.epsilon_at(0, CFG) == pytest.approx(1.0)
    assert trainer.epsilon_at(CFG["eps_decay_episodes"], CFG) == pytest.approx(0.05)
    assert trainer.epsilon_at(10**9, CFG) == pytest.approx(0.05)
    assert trainer.lr_at(0, CFG) == pytest.approx(1e-3)
    assert trainer.lr_at(trainer.total_episodes(CFG), CFG) == pytest.approx(1e-4)


# ---------- học được ----------

def test_learn_step_reduces_loss_on_fixed_data():
    cfg = dict(batch_size=32, gamma=0.99, grad_clip=10.0)
    torch.manual_seed(0)
    net = QNet(channels=8, hidden=16)
    target = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    rb = ReplayBuffer(200, seed=0)
    rng = random.Random(0)
    for _ in range(200):  # dữ liệu cố định: thưởng +1 khi thả cột 3 và kết thúc, ngược lại 0 và kết thúc
        s = np.zeros((2, ROWS, COLS), np.uint8)
        s[0, 0, rng.randrange(COLS)] = 1
        a = rng.randrange(COLS)
        rb.add(s, a, 1.0 if a == 3 else 0.0, s, True, np.zeros(COLS, bool))
    losses = [trainer.learn_step(net, target, opt, rb, cfg, "cpu") for _ in range(150)]
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20])


# ---------- vòng huấn luyện nhỏ (CPU) ----------

def tiny_cfg():
    cfg = trainer.load_config("configs/rl_smoke.json")
    cfg["device"] = "cpu"
    cfg["stages"] = [
        {"name": "vs_random", "episodes": 12, "opponent": "random"},
        {"name": "vs_heuristic", "episodes": 6, "opponent": "heuristic", "opp_eps": 0.1},
        {"name": "self_pool", "episodes": 6, "opponent": "self_pool", "pool_prob": 0.5, "pool_every": 3,
         "pool_size": 2, "opp_eps": 0.05},
    ]
    cfg.update(min_replay=30, batch_size=16, eps_decay_episodes=12, target_update_steps=5)
    cfg["eval"] = {"every": 12, "games": 4, "dev_seed_start": 0}
    return cfg


def test_run_training_writes_logs_and_checkpoints(tmp_path):
    meta = trainer.run_training(tiny_cfg(), "RL-T", logs_root=tmp_path / "logs", ckpt_root=tmp_path / "ck", verbose=False)
    logs, ck = tmp_path / "logs" / "RL-T", tmp_path / "ck" / "RL-T"
    for f in ("train_log.csv", "config.json", "meta.json"):
        assert (logs / f).exists()
    for f in ("last.pt", "best.pt"):
        assert (ck / f).exists()
    rows = list(csv.DictReader(open(logs / "train_log.csv", encoding="utf-8")))
    assert [r["episode"] for r in rows] == ["12", "24"]
    assert all(0.0 <= float(r["dev_random_score"]) <= 1.0 for r in rows)
    m = json.loads((logs / "meta.json").read_text(encoding="utf-8"))
    assert m["episodes_done"] == 24 and m["stopped_by"] == "completed" and m["updates"] > 0
    assert meta["best_checkpoint"]["file"] == "best.pt"
    assert isinstance(load_network(str(ck / "best.pt"), "cpu"), QNet)


def test_run_training_refuses_to_overwrite(tmp_path):
    kw = dict(logs_root=tmp_path / "logs", ckpt_root=tmp_path / "ck", verbose=False)
    trainer.run_training(tiny_cfg(), "RL-T", **kw)
    with pytest.raises(FileExistsError):
        trainer.run_training(tiny_cfg(), "RL-T", **kw)


def test_run_training_stops_on_time_limit_and_still_saves(tmp_path):
    meta = trainer.run_training(tiny_cfg(), "RL-T", max_minutes=0.0, logs_root=tmp_path / "logs",
                                ckpt_root=tmp_path / "ck", verbose=False)
    assert meta["stopped_by"] == "max_minutes" and meta["episodes_done"] == 1
    assert (tmp_path / "ck" / "RL-T" / "last.pt").exists()


def test_run_training_init_from_loads_weights_and_checks_architecture(tmp_path):
    cfg = tiny_cfg()
    net = QNet(**cfg["network"])
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(0.123)
    src = tmp_path / "src.pt"
    save_checkpoint(src, net, cfg["network"], 5)
    cfg["init_from"] = str(src)
    seen = {}
    orig = trainer.net_policy

    def spy(n, device, eps, rng):
        seen.setdefault("first", [p.detach().cpu().clone() for p in n.parameters()])
        return orig(n, device, eps, rng)
    trainer.net_policy = spy
    try:
        trainer.run_training(cfg, "RL-W", logs_root=tmp_path / "logs", ckpt_root=tmp_path / "ck", verbose=False)
    finally:
        trainer.net_policy = orig
    assert all(torch.allclose(p, torch.full_like(p, 0.123)) for p in seen["first"])  # bắt đầu từ trọng số đã nạp
    bad = tiny_cfg()
    bad["network"] = {"channels": 4, "hidden": 8}
    bad["init_from"] = str(src)
    with pytest.raises(ValueError):
        trainer.run_training(bad, "RL-W2", logs_root=tmp_path / "logs", ckpt_root=tmp_path / "ck", verbose=False)


# ---------- tích hợp arena ----------

def test_arena_builds_dqn_from_spec(tmp_path, monkeypatch):
    from agents import dqn_agent
    monkeypatch.setattr(dqn_agent, "CHECKPOINT_DIR", tmp_path)
    save_checkpoint(tmp_path / "RL-X" / "best.pt", QNet(4, 8), {"channels": 4, "hidden": 8}, 1)
    ag = tournament.make_agent("dqn_RL-X_best", 0)
    assert ag.name == "DQN-RL-X-best"
    assert ag.choose_action(board.Board()) in range(COLS)


# ---------- đánh giá dev ----------

def test_evaluate_matches_independent_count():
    # Đếm lại bằng cách tự chơi từng ván và so sánh đối tượng thắng, không dùng arena.match.play_game.
    torch.manual_seed(1)
    net = QNet(channels=8, hidden=16)
    got = trainer.evaluate(net, "cpu", "random", 20, 0)
    w = d = l = 0
    for i in range(20):
        me, opp = DQNAgent(net), RandomAgent(seed=i)
        agents = {P1: me, P2: opp} if i % 2 == 0 else {P1: opp, P2: me}
        b = board.Board()
        while not b.is_terminal():
            b.step(agents[b.to_move].choose_action(b))
        if b.winner is None:
            d += 1
        elif agents[b.winner] is me:
            w += 1
        else:
            l += 1
    assert (got["win"], got["draw"], got["loss"]) == (w, d, l)
    assert got["score"] == pytest.approx((w + 0.5 * d) / 20)

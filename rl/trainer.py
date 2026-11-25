import argparse
import copy
import csv
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.dqn_agent import DQNAgent, greedy_action
from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from arena.match import play_game
from arena.metrics import score
from engine.board import Board, PLAYER_1, PLAYER_2
from engine.encoding import encode, legal_mask
from rl.checkpoint import CHECKPOINT_DIR, save_checkpoint
from rl.network import QNet
from rl.replay import ReplayBuffer

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
LOG_FIELDS = ["episode", "stage", "epsilon", "lr", "loss", "replay_size", "updates",
              "train_win", "train_draw", "train_loss", "elapsed_s",
              "dev_random_score", "dev_heuristic_score"]


# ---------- cấu hình và lịch ----------

def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_device(cfg):
    want = cfg.get("device", "auto")
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def total_episodes(cfg):
    return sum(s["episodes"] for s in cfg["stages"])


def stage_at(episode, stages):
    # Giai đoạn chứa episode (đếm từ 0). Quá tổng thì trả giai đoạn cuối.
    end = 0
    for st in stages:
        end += st["episodes"]
        if episode < end:
            return st
    return stages[-1]


def epsilon_at(episode, cfg):
    frac = min(1.0, episode / max(1, cfg["eps_decay_episodes"]))
    return cfg["eps_start"] + (cfg["eps_end"] - cfg["eps_start"]) * frac


def lr_at(episode, cfg):
    frac = min(1.0, episode / max(1, total_episodes(cfg)))
    return cfg["lr_start"] + (cfg["lr_end"] - cfg["lr_start"]) * frac


# ---------- chính sách (hàm nhận board, trả cột) ----------

def net_policy(net, device, eps, rng):
    def policy(board):
        if eps > 0 and rng.random() < eps:
            return rng.choice(board.legal_actions())
        return greedy_action(net, board, device)
    return policy


def random_policy(rng):
    return lambda board: rng.choice(board.legal_actions())


def agent_policy(agent, eps, rng):
    def policy(board):
        if eps > 0 and rng.random() < eps:
            return rng.choice(board.legal_actions())
        return agent.choose_action(board)
    return policy


def make_opponent(stage, net, pool, heuristic, device, rng):
    kind = stage["opponent"]
    if kind == "random":
        return random_policy(rng)
    if kind == "heuristic":
        return agent_policy(heuristic, stage.get("opp_eps", 0.0), rng)
    if kind in ("self", "self_pool"):
        chosen = net
        if kind == "self_pool" and pool and rng.random() < stage.get("pool_prob", 0.5):
            chosen = rng.choice(pool)
        return net_policy(chosen, device, stage.get("opp_eps", 0.0), rng)
    raise ValueError(f"Đối thủ không hợp lệ: {kind}")


# ---------- chơi một ván và ghi kinh nghiệm ----------

def play_episode(policies, replay):
    # policies: {PLAYER_1: f, PLAYER_2: f}. Ghi MỌI nước (cả hai bên), mỗi nước theo góc nhìn người vừa đi.
    # Trả về (winner hoặc None nếu hòa, số nước).
    board = Board()
    plies = 0
    while not board.is_terminal():
        mover = board.to_move
        state = encode(board)
        action = policies[mover](board)
        board.step(action)
        done = board.is_terminal()
        reward = 1.0 if (done and board.winner == mover) else 0.0
        replay.add(state, action, reward, encode(board), done, legal_mask(board))
        plies += 1
    return board.winner, plies


# ---------- học ----------

def compute_targets(online, target, rewards, dones, next_states, next_masks, gamma):
    # Double DQN + negamax. Cột đầy bị loại khỏi argmax. Ván đã kết thúc thì mục tiêu chính là phần thưởng.
    with torch.no_grad():
        q_online = online(next_states).masked_fill(~next_masks, float("-inf"))
        best = q_online.argmax(dim=1, keepdim=True)
        q_next = target(next_states).gather(1, best).squeeze(1)
        return rewards + (1.0 - dones) * (-gamma) * q_next


def learn_step(online, target, optimizer, replay, cfg, device):
    s, a, r, ns, d, nm = replay.sample(cfg["batch_size"], device)
    y = compute_targets(online, target, r, d, ns, nm, cfg["gamma"])
    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(q, y)  # Huber
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.get("grad_clip", 10.0))
    optimizer.step()
    return float(loss.item())


# ---------- đánh giá trên ván dev ----------

def evaluate(net, device, opponent, games, seed_start):
    # Mạng greedy đấu opponent ("random" hoặc "heuristic"), chia đều đi trước/sau, dải seed dev.
    wins = draws = losses = 0
    for i in range(games):
        seed = seed_start + i
        opp = RandomAgent(seed=seed) if opponent == "random" else HeuristicAgent(seed=seed)
        me = DQNAgent(net, device=device)
        me_first = i % 2 == 0
        winner, _ = play_game(me, opp) if me_first else play_game(opp, me)
        if winner is None:
            draws += 1
        elif (winner == PLAYER_1) == me_first:
            wins += 1
        else:
            losses += 1
    return dict(win=wins, draw=draws, loss=losses, score=score(wins, draws, games))


# ---------- vòng huấn luyện ----------

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_training(cfg, run_name, config_path=None, max_minutes=None, logs_root=LOGS,
                 ckpt_root=CHECKPOINT_DIR, verbose=True):
    run_logs, run_ckpt = Path(logs_root) / run_name, Path(ckpt_root) / run_name
    if run_logs.exists() or run_ckpt.exists():
        raise FileExistsError(f"Không ghi đè: {run_logs} hoặc {run_ckpt} đã tồn tại. Dùng tên run mới.")
    run_logs.mkdir(parents=True)
    run_ckpt.mkdir(parents=True)
    (run_logs / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    device = resolve_device(cfg)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    net = QNet(**cfg["network"]).to(device)
    pool = []
    if cfg.get("init_from"):
        # Khởi động ấm (warm start): chỉ nạp TRỌNG SỐ từ checkpoint cũ. Bộ nhớ kinh nghiệm, trạng thái Adam, mạng đích
        # và kho đối thủ không có trong checkpoint nên bắt đầu lại; bản gốc của mạng được đưa vào kho đối thủ.
        data = torch.load(ROOT / cfg["init_from"], map_location=device)
        if dict(data["network"]) != dict(cfg["network"]):
            raise ValueError(f"Kiến trúc trong {cfg['init_from']} khác cấu hình: {data['network']} vs {cfg['network']}")
        net.load_state_dict(data["state_dict"])
        base = copy.deepcopy(net)
        base.eval()
        pool.append(base)
    target = copy.deepcopy(net)
    target.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr_start"])
    replay = ReplayBuffer(cfg["replay_capacity"], seed=seed + 1)
    heuristic = HeuristicAgent(seed=seed + 2)
    total = total_episodes(cfg)
    ev = cfg["eval"]

    t0 = time.perf_counter()
    updates = 0
    loss_sum, loss_n = 0.0, 0
    tw = td = tl = 0
    best_score = -1.0
    best_episode = 0
    stopped_by = "completed"
    last_eval = {}
    log_file = open(run_logs / "train_log.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    writer.writeheader()

    def do_eval(done_ep, stage_name, eps, lr):
        nonlocal loss_sum, loss_n, tw, td, tl, best_score, best_episode, last_eval
        r = evaluate(net, device, "random", ev["games"], ev["dev_seed_start"])
        h = evaluate(net, device, "heuristic", ev["games"], ev["dev_seed_start"])
        last_eval = dict(random=r, heuristic=h)
        writer.writerow(dict(
            episode=done_ep, stage=stage_name, epsilon=round(eps, 4), lr=lr,
            loss=round(loss_sum / loss_n, 6) if loss_n else "", replay_size=len(replay), updates=updates,
            train_win=tw, train_draw=td, train_loss=tl, elapsed_s=round(time.perf_counter() - t0, 1),
            dev_random_score=round(r["score"], 4), dev_heuristic_score=round(h["score"], 4)))
        log_file.flush()
        crit = (r["score"] + h["score"]) / 2  # tiêu chí chọn checkpoint: trung bình score dev vs Random và Heuristic
        extra = dict(dev_random=r["score"], dev_heuristic=h["score"])
        save_checkpoint(run_ckpt / "last.pt", net, cfg["network"], done_ep, extra)
        if crit > best_score:
            best_score, best_episode = crit, done_ep
            save_checkpoint(run_ckpt / "best.pt", net, cfg["network"], done_ep, extra)
        if verbose:
            print(f"[ep {done_ep}/{total}] {stage_name} eps={eps:.3f} loss={loss_sum / max(1, loss_n):.4f} "
                  f"dev: random {r['score']:.3f}, heuristic {h['score']:.3f} | {time.perf_counter() - t0:.0f}s", flush=True)
        loss_sum, loss_n, tw, td, tl = 0.0, 0, 0, 0, 0

    eps = lr = 0.0
    stage = cfg["stages"][0]
    done_ep = 0
    for ep in range(total):
        stage = stage_at(ep, cfg["stages"])
        eps = epsilon_at(ep, cfg)
        lr = lr_at(ep, cfg)
        learner_first = ep % 2 == 0
        learner = net_policy(net, device, eps, rng)
        opponent = make_opponent(stage, net, pool, heuristic, device, rng)
        policies = {PLAYER_1: learner, PLAYER_2: opponent} if learner_first else {PLAYER_1: opponent, PLAYER_2: learner}
        winner, _ = play_episode(policies, replay)
        me = PLAYER_1 if learner_first else PLAYER_2
        if winner is None:
            td += 1
        elif winner == me:
            tw += 1
        else:
            tl += 1

        if len(replay) >= cfg["min_replay"]:
            for g in optimizer.param_groups:
                g["lr"] = lr
            for _ in range(cfg["updates_per_episode"]):
                loss_sum += learn_step(net, target, optimizer, replay, cfg, device)
                loss_n += 1
                updates += 1
                if updates % cfg["target_update_steps"] == 0:
                    target.load_state_dict(net.state_dict())

        if stage["opponent"] == "self_pool" and (ep + 1) % stage.get("pool_every", 5000) == 0:
            snap = copy.deepcopy(net)
            snap.eval()
            pool.append(snap)
            del pool[:-stage.get("pool_size", 5)]

        done_ep = ep + 1
        out_of_time = max_minutes is not None and (time.perf_counter() - t0) / 60 >= max_minutes
        if done_ep % ev["every"] == 0 or done_ep == total or out_of_time:
            do_eval(done_ep, stage["name"], eps, lr)
        if cfg.get("snapshot_every") and done_ep % cfg["snapshot_every"] == 0:
            save_checkpoint(run_ckpt / f"ep{done_ep}.pt", net, cfg["network"], done_ep)
        if out_of_time:
            stopped_by = "max_minutes"
            break
    log_file.close()

    elapsed = time.perf_counter() - t0
    meta = dict(
        run=run_name, config_file=str(config_path) if config_path else None, seed=seed, device=device,
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
        episodes_done=done_ep, episodes_planned=total, updates=updates, replay_size=len(replay),
        elapsed_seconds=round(elapsed, 1), stopped_by=stopped_by,
        best_checkpoint=dict(file="best.pt", episode=best_episode, criterion="mean dev score vs Random and Heuristic",
                             value=round(best_score, 4)),
        last_dev_eval=last_eval, torch=torch.__version__, python=platform.python_version(),
        machine=platform.platform(), processor=platform.processor(), git_commit=git_commit(),
        note="Ván dev dùng dải seed dev_seed_start trở đi; dải test không dùng. Huấn luyện trên GPU không tái lập "
             "chính xác từng bit. Chưa có chức năng resume.")
    (run_logs / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Huấn luyện Double DQN cho Connect Four")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="tên lần chạy, ví dụ RL-001 (không được trùng)")
    ap.add_argument("--max-minutes", type=float, default=None, help="dừng sau số phút này (vẫn lưu checkpoint)")
    args = ap.parse_args()
    meta = run_training(load_config(args.config), args.run, config_path=args.config, max_minutes=args.max_minutes)
    print(json.dumps({k: meta[k] for k in ("run", "episodes_done", "elapsed_seconds", "stopped_by", "best_checkpoint")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

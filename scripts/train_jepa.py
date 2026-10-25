"""Train the MindJEPA terminal/physical world model.

Data options:
  1. Synthetic virtual shell sessions (default) - learnable dynamics with
     realistic error/recovery patterns (see terminal_world.py)
  2. Real shell history via --jsonl path/to/sessions.jsonl
     (rows: {"session_id": 0, "cmd": "...", "output": "...", "exit": 0})
  3. Physical I/O via --physio sensors.npy motors.npy (arrays, T x D)

The trained model is saved to checkpoints/jepa.pt and can drive the coupled
mind-world knot loop (scripts/run_simulation.py --mode world_knot).

Usage:
    python scripts/train_jepa.py --epochs 40 --sessions 256
    python scripts/train_jepa.py --jsonl my_history.jsonl --epochs 80
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from mind_diffusion.mind_jepa_bridge import (
    MindJEPA, TerminalWorldAdapter, rankme, action_gap, rollout_error,
)
from mind_diffusion.terminal_world import generate_sessions, load_jsonl_sessions

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s | %(message)s")
log = logging.getLogger("train_jepa")


def sessions_to_batch(sessions, adapter, max_turns=20):
    """Pad sessions to (B, T, D) tensors + mask. Returns obs, act, mask."""
    seqs = [adapter.session_to_frames(s) for s in sessions]
    T = min(max(s[0].shape[0] for s in seqs), max_turns)
    T = max(T, 3)
    D_obs, D_act = seqs[0][0].shape[1], seqs[0][1].shape[1]
    B = len(seqs)
    obs = np.zeros((B, T, D_obs), dtype=np.float32)
    act = np.zeros((B, T, D_act), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i, (o, a) in enumerate(seqs):
        n = min(o.shape[0], T)
        obs[i, :n], act[i, :n], mask[i, :n] = o[:n], a[:n], 1.0
    return torch.from_numpy(obs), torch.from_numpy(act), torch.from_numpy(mask)


def masked_loss(jepa, obs, act, mask):
    """JEPA loss with per-session masking - simply trims to valid lengths.

    (Training on padded zeros would teach the model fake transitions, so we
    run each session truncated to its true length via the mask.)
    """
    total = None
    parts = {"pred": 0.0, "self": 0.0, "reg": 0.0}
    n = 0
    for b in range(obs.shape[0]):
        L = int(mask[b].sum())
        if L < 3:
            continue
        o, a = obs[b:b + 1, :L], act[b:b + 1, :L]
        loss, pl, sl, rl = jepa.training_step(o, a)
        total = loss if total is None else total + loss
        parts["pred"] += pl
        parts["self"] += sl
        parts["reg"] += rl
        n += 1
    if total is None:
        return None, None
    total = total / n
    parts = {k: v / n for k, v in parts.items()}
    parts["total"] = float(total)
    return total, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sessions", type=int, default=256,
                    help="number of synthetic sessions")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--obs-dim", type=int, default=64)
    ap.add_argument("--act-dim", type=int, default=32)
    ap.add_argument("--zdim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jsonl", type=str, default=None,
                    help="real sessions JSONL instead of synthetic")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", type=str, default="checkpoints/jepa.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- data --------------------------------------------------------------- #
    if args.jsonl:
        sessions = load_jsonl_sessions(args.jsonl)
    else:
        log.info("generating %d synthetic terminal sessions...", args.sessions)
        sessions = generate_sessions(n=args.sessions, seed=args.seed)
    obs_all, act_all, mask_all = sessions_to_batch(sessions, TerminalWorldAdapter(
        args.obs_dim, args.act_dim))
    N = obs_all.shape[0]
    n_val = max(1, int(N * args.val_frac))
    perm = torch.randperm(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    log.info("data: %d train / %d val sessions, %d turns max",
             len(tr_idx), len(val_idx), obs_all.shape[1])

    # ---- model -------------------------------------------------------------- #
    jepa = MindJEPA(obs_dim=args.obs_dim, act_dim=args.act_dim,
                    zdim=args.zdim, hidden=args.hidden)
    params = sum(p.numel() for p in jepa.parameters())
    log.info("MindJEPA: %d params", params)
    opt = torch.optim.AdamW(jepa.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    best_val = float("inf")
    for epoch in range(args.epochs):
        jepa.train()
        perm_t = tr_idx[torch.randperm(len(tr_idx))]
        ep = {"total": 0.0, "pred": 0.0, "self": 0.0, "reg": 0.0}
        nb = 0
        for i in range(0, len(perm_t), args.batch):
            idx = perm_t[i:i + args.batch]
            loss, parts = masked_loss(jepa, obs_all[idx], act_all[idx], mask_all[idx])
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), 1.0)
            opt.step()
            for k in ep:
                ep[k] += parts[k]
            nb += 1
        sched.step()

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            jepa.eval()
            with torch.no_grad():
                _, vparts = masked_loss(jepa, obs_all[val_idx], act_all[val_idx],
                                        mask_all[val_idx])
                vloss = vparts["total"]
                vo, va = obs_all[val_idx], act_all[val_idx]
                gap, et, es = action_gap(jepa, vo, va)
                errs = rollout_error(jepa, vo, va, horizons=(1, 4, 8))
                rk = rankme(jepa.frame_enc(vo.reshape(-1, vo.shape[-1])))
            # save on usefulness: low val loss AND high action gap
            score = vloss - max(gap, 0.0)
            mark = ""
            if score < best_val:
                best_val = score
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(jepa.state_dict(), args.out)
                mark = "  <- saved"
            log.info("[ep %3d/%d] train=%.4f val=%.4f | pred=%.4f self=%.4f reg=%.4f "
                     "| gap=%.4f (true=%.3f shuf=%.3f) | rollout=%s | rankme=%.1f%s",
                     epoch + 1, args.epochs, ep["total"] / max(nb, 1), vloss,
                     vparts["pred"], vparts["self"], vparts["reg"],
                     gap, et, es,
                     {k: round(v, 3) for k, v in errs.items()}, rk, mark)

    log.info("done in %.1fs. best val=%.4f", time.time() - t0, best_val)
    log.info("saved -> %s", args.out)


if __name__ == "__main__":
    main()

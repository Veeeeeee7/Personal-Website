"""TD(0) learning over afterstates — Szubert & Jaśkowski (2014).

The idea in one paragraph: an *afterstate* is the board right after your
slide and merge, before the random tile drops. It is a deterministic
consequence of your move, so a value function defined on afterstates
never has to average over where the next tile lands. That removes the
stochasticity from the learning target entirely, which is why this
converges where naive state-value TD flounders.

    V(s'_t)  <-  V(s'_t) + alpha * [ r_{t+1} + V(s'_{t+1}) - V(s'_t) ]

where s'_t is the afterstate you chose and s'_{t+1} is the afterstate of
the greedy move from the state that followed it.

Usage:
    python train.py --episodes 200000
    python train.py --episodes 500000 --config large --alpha 0.05
    python train.py --resume checkpoints/web-5tuple.npz --episodes 100000
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np

from engine import (
    ALL_DIRS,
    WIN_EXPONENT,
    face_value,
    is_game_over,
    move,
    new_board,
    spawn_tile,
)
from ntuple import CONFIGS, NTupleNetwork


def greedy(net: NTupleNetwork, board: np.ndarray):
    """Best move by immediate reward + learned afterstate value.

    Returns (direction, reward, afterstate) or None if no move is legal.
    """
    best = None
    best_score = -np.inf

    for direction in ALL_DIRS:
        after, reward, moved = move(board, direction)
        if not moved:
            continue
        score = reward + net.value(after)
        if score > best_score:
            best_score = score
            best = (direction, reward, after)

    return best


def play_episode(
    net: NTupleNetwork,
    rng: np.random.Generator,
    alpha: float,
    target_exponent: int,
    learn: bool = True,
):
    """One game. Returns (score, max_tile, moves, won)."""
    board = new_board(rng)
    score = 0
    moves = 0

    prev_afterstate = None

    while True:
        chosen = greedy(net, board)
        if chosen is None:
            break

        direction, reward, afterstate = chosen

        # Update the PREVIOUS afterstate, now that we know what followed
        # it: the reward this move earned, plus the value of the
        # afterstate it led to. The update always runs one step behind
        # because r_{t+1} isn't known until move t+1 is chosen.
        #
        #   V(s'_t) <- V(s'_t) + alpha * [ r_{t+1} + V(s'_{t+1}) - V(s'_t) ]
        if learn and prev_afterstate is not None:
            td_target = reward + net.value(afterstate)
            net.update(prev_afterstate, alpha * (td_target - net.value(prev_afterstate)))

        prev_afterstate = afterstate

        score += reward
        moves += 1

        board = afterstate.copy()

        # Stop at the target rather than playing on. The deployed agent
        # restarts at 2048, so training past it would optimise for a game
        # the browser never plays — and would need a bigger alphabet.
        if board.max() >= target_exponent:
            if learn:
                # Terminal: nothing follows, so the target is just 0.
                net.update(prev_afterstate, alpha * (0.0 - net.value(prev_afterstate)))
            return score, int(board.max()), moves, True

        if not spawn_tile(board, rng):
            break
        if is_game_over(board):
            break

    if learn and prev_afterstate is not None:
        net.update(prev_afterstate, alpha * (0.0 - net.value(prev_afterstate)))

    return score, int(board.max()), moves, False


def evaluate(net: NTupleNetwork, rng, games: int, target: int):
    scores, wins, tiles = [], 0, []
    for _ in range(games):
        score, tile, _, won = play_episode(net, rng, 0.0, target, learn=False)
        scores.append(score)
        tiles.append(tile)
        wins += int(won)
    return float(np.mean(scores)), wins / games, int(np.max(tiles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--config", choices=list(CONFIGS), default="web-5tuple")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Learning rate. 0.1 is a good start; anneal if it plateaus.")
    parser.add_argument("--target", type=int, default=WIN_EXPONENT,
                        help="Stop an episode at this exponent. 11 = the 2048 tile.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out", type=str, default="checkpoints")
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.resume:
        net = NTupleNetwork.load(args.resume)
        print(f"resumed from {args.resume}")
    else:
        net = NTupleNetwork(CONFIGS[args.config])

    print(net.config.describe())
    print(f"lookup paths per evaluation: {net.n_paths}")
    print()

    os.makedirs(args.out, exist_ok=True)
    checkpoint = os.path.join(args.out, f"{net.config.name}.npz")

    recent_scores = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)
    started = time.time()

    for episode in range(1, args.episodes + 1):
        score, tile, moves, won = play_episode(net, rng, args.alpha, args.target)
        recent_scores.append(score)
        recent_wins.append(won)

        if episode % args.log_every == 0:
            elapsed = time.time() - started
            print(
                f"ep {episode:>8,}  "
                f"mean {np.mean(recent_scores):>8.0f}  "
                f"win {np.mean(recent_wins):>5.1%}  "
                f"best {face_value(tile):>5}  "
                f"{episode / elapsed:>5.0f} ep/s",
                flush=True,
            )

        if episode % args.save_every == 0:
            net.save(checkpoint)

        if episode % args.eval_every == 0:
            mean, win_rate, best_tile = evaluate(net, rng, 100, args.target)
            print(
                f"  eval: mean {mean:.0f}  win {win_rate:.1%}  best {face_value(best_tile)}"
            )

    net.save(checkpoint)
    print(f"\nsaved {checkpoint}")
    print("next: python export.py --checkpoint", checkpoint)


if __name__ == "__main__":
    main()

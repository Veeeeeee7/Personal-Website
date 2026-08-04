"""Cross-check the Python side against the TypeScript side.

The trainer learns against engine.py; the browser plays against
engine.ts. If they disagree about even one merge, the deployed policy is
being asked questions it was never trained on — and every offline metric
will still look perfect. Same for the weights: if the browser unpacks
the tables in a different order than the trainer packed them, nothing
errors, the agent just quietly plays like it never learned anything.

So: generate fixtures from Python, replay them in Node, assert equality.

Usage:
    python verify.py                          # engine fixtures
    npm run verify:engine

    python verify.py --policy public/policy   # policy fixtures
    npm run verify:policy
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from engine import (
    ALL_DIRS,
    CELL_INDEX,
    SYMMETRIES,
    is_game_over,
    move,
    new_board,
    spawn_tile,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures.json")
POLICY_FIXTURES = os.path.join(HERE, "policy_fixtures.json")


def collect_boards(count: int, seed: int = 7) -> list[np.ndarray]:
    """Boards from real rollouts, not uniform noise.

    Random 16-cell arrays are trivially easy to get right and never look
    like a real position. Rollout boards exercise the cases that actually
    occur: long merge chains, full boards, near-death positions.
    """
    rng = np.random.default_rng(seed)
    boards: list[np.ndarray] = []

    while len(boards) < count:
        board = new_board(rng)
        while len(boards) < count:
            boards.append(board.copy())

            legal = [d for d in ALL_DIRS if move(board, d)[2]]
            if not legal:
                break
            after, _, _ = move(board, rng.choice(legal))
            board = after.copy()
            if not spawn_tile(board, rng):
                break
            if is_game_over(board):
                break

    return boards


def write_policy_fixtures(policy_dir: str) -> None:
    """Evaluate an exported policy on real boards, the way the browser will.

    Deliberately reads back the *quantized* weights.bin rather than the
    float checkpoint: the point is to test the bytes that actually ship,
    including the manifest's scale, not the network in memory.
    """
    manifest_path = os.path.join(policy_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    dtype = {"int16": "<i2", "int32": "<i4", "float32": "<f4"}[manifest["quant"]["dtype"]]
    weights_path = os.path.join(policy_dir, manifest["weights"])
    stored = np.fromfile(weights_path, dtype=dtype)
    scale = manifest["quant"]["scale"]

    alphabet = manifest["alphabet"]
    tuples = manifest["tuples"]
    syms = SYMMETRIES if manifest["symmetric"] else [SYMMETRIES[0]]

    # Rebuild the lookup plan straight from the manifest — not from
    # ntuple.py. If ntuple.py and the manifest ever disagree, this
    # catches it; sharing the code would hide it.
    offsets, cursor = [], 0
    for t in tuples:
        offsets.append(cursor)
        cursor += alphabet ** len(t)
    assert cursor == stored.size, f"manifest wants {cursor} weights, bin has {stored.size}"

    paths = []
    for t_index, tuple_cells in enumerate(tuples):
        place = [alphabet**k for k in range(len(tuple_cells))]
        for sym in syms:
            paths.append(([int(sym[c]) for c in tuple_cells], place, offsets[t_index]))

    def value(board: np.ndarray) -> float:
        total = 0
        for cells, place, offset in paths:
            index = sum(min(int(board[c]), alphabet - 1) * m for c, m in zip(cells, place))
            total += int(stored[offset + index])
        return float(total * scale)

    def best_dir(board: np.ndarray):
        best, best_score = None, -np.inf
        for direction in ALL_DIRS:
            after, reward, moved = move(board, direction)
            if not moved:
                continue
            score = reward + value(after)
            if score > best_score:
                best_score, best = score, direction
        return best

    cases = []
    for board in collect_boards(300, seed=99):
        cases.append(
            {
                "board": [int(v) for v in board],
                "value": value(board),
                "bestDir": best_dir(board),
            }
        )

    payload = {
        "manifest": manifest,
        "weightsPath": os.path.relpath(weights_path, os.path.join(HERE, "..")),
        "cases": cases,
    }
    with open(POLICY_FIXTURES, "w") as f:
        json.dump(payload, f)

    print(f"wrote {POLICY_FIXTURES}: {len(cases)} boards from {policy_dir}")
    print("now run: npm run verify:policy")


def write_engine_fixtures() -> None:
    boards = collect_boards(4000)

    cases = []
    for board in boards:
        for direction in ALL_DIRS:
            after, reward, moved = move(board, direction)
            cases.append(
                {
                    "board": [int(v) for v in board],
                    "dir": int(direction),
                    "after": [int(v) for v in after],
                    "reward": int(reward),
                    "moved": bool(moved),
                }
            )

    # Some deliberately nasty hand-built lines that rollouts rarely hit.
    edge_cases = [
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # double merge in one row
        [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # two distinct merges
        [2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # merge in the middle
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # merge across a gap
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # three-of-a-kind: only one merge
        [15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # at the exponent ceiling
        [1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1],  # full, no legal move
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1],  # full, no merges
    ]
    for cells in edge_cases:
        board = np.array(cells, dtype=np.uint8)
        for direction in ALL_DIRS:
            after, reward, moved = move(board, direction)
            cases.append(
                {
                    "board": cells,
                    "dir": int(direction),
                    "after": [int(v) for v in after],
                    "reward": int(reward),
                    "moved": bool(moved),
                }
            )

    payload = {
        "cases": cases,
        "symmetries": [[int(v) for v in s] for s in SYMMETRIES],
        "cellIndex": [[[int(v) for v in row] for row in d] for d in CELL_INDEX],
    }

    with open(FIXTURES, "w") as f:
        json.dump(payload, f)

    print(f"wrote {FIXTURES}: {len(cases):,} move cases, {len(SYMMETRIES)} symmetries")
    print("now run: npm run verify:engine")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        nargs="?",
        const=os.path.join(HERE, "..", "public", "policy"),
        default=None,
        help="Write policy fixtures from an exported policy dir instead of engine fixtures.",
    )
    args = parser.parse_args()

    if args.policy:
        write_policy_fixtures(os.path.abspath(args.policy))
    else:
        write_engine_fixtures()


if __name__ == "__main__":
    main()

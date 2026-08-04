"""2048 engine — the Python mirror of src/game/engine.ts.

These two files MUST stay in agreement. A trainer that merges slightly
differently from the browser produces a policy that is subtly wrong in
deployment and looks fine in every offline metric you'd think to check.
`verify.py` cross-checks them on random rollouts; run it after touching
either side.

Board is a length-16 uint8 array of EXPONENTS (0 = empty, 1 = "2",
11 = "2048"), row-major:

    0  1  2  3
    4  5  6  7
    8  9 10 11
   12 13 14 15
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # numba is optional; without it training is ~50x slower
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap


SIZE = 4
CELLS = 16
WIN_EXPONENT = 11
MAX_EXPONENT = 15

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ALL_DIRS = (UP, RIGHT, DOWN, LEFT)
DIR_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}


# ── Cell addressing ──────────────────────────────────────────────
# Each direction is 4 lines of 4 positions, with position 0 at the edge
# tiles slide toward. One slide-toward-zero routine then covers all four.

def _build_cell_index() -> np.ndarray:
    table = np.zeros((4, 4, 4), dtype=np.uint8)  # [dir][line][pos]
    for line in range(4):
        for pos in range(4):
            table[LEFT, line, pos] = line * 4 + pos
            table[RIGHT, line, pos] = line * 4 + (3 - pos)
            table[UP, line, pos] = pos * 4 + line
            table[DOWN, line, pos] = (3 - pos) * 4 + line
    return table


CELL_INDEX = _build_cell_index()


# ── Row lookup tables ────────────────────────────────────────────
# A line is 4 nibbles: key = p0 | p1<<4 | p2<<8 | p3<<12.
# All 65536 lines are precomputed once at import (~1s pure Python).

def _build_line_tables() -> tuple[np.ndarray, np.ndarray]:
    packed = np.zeros(65536, dtype=np.uint16)
    reward = np.zeros(65536, dtype=np.int32)

    for key in range(65536):
        src = [(key >> (4 * p)) & 0xF for p in range(4)]
        filled = [p for p in range(4) if src[p] != 0]

        out: list[int] = []
        gained = 0
        i = 0
        while i < len(filled):
            v = src[filled[i]]
            nxt = filled[i + 1] if i + 1 < len(filled) else -1
            # One merge per tile per move; i += 2 enforces that.
            if nxt != -1 and src[nxt] == v and v < MAX_EXPONENT:
                out.append(v + 1)
                gained += 1 << (v + 1)
                i += 2
            else:
                out.append(v)
                i += 1

        result = 0
        for p, v in enumerate(out):
            result |= v << (4 * p)

        packed[key] = result
        reward[key] = gained

    return packed, reward


LINE_PACKED, LINE_REWARD = _build_line_tables()


# ── Move ─────────────────────────────────────────────────────────

@njit(cache=True)
def _move(board, dir_cells, line_packed, line_reward):
    """Returns (afterstate, reward, moved). Does not mutate `board`."""
    out = np.zeros(16, dtype=np.uint8)
    total = 0
    moved = False

    for line in range(4):
        # Widen before shifting. `board` is uint8, and numpy shifts stay
        # in the input dtype — so `board[i] << 12` silently overflows to
        # zero and packs a corrupt key. Cast first, always.
        key = np.int64(0)
        for pos in range(4):
            key |= np.int64(board[dir_cells[line, pos]]) << (4 * pos)

        result = np.int64(line_packed[key])
        total += line_reward[key]
        if result != key:
            moved = True

        for pos in range(4):
            out[dir_cells[line, pos]] = (result >> (4 * pos)) & 0xF

    if not moved:
        return board.copy(), 0, False
    return out, total, True


def move(board: np.ndarray, direction: int) -> tuple[np.ndarray, int, bool]:
    """Apply a move.

    Returns the AFTERSTATE — post-slide, PRE-spawn. That separation is
    the whole basis of afterstate TD learning, so don't collapse it.
    """
    return _move(board, CELL_INDEX[direction], LINE_PACKED, LINE_REWARD)


# ── Spawning ─────────────────────────────────────────────────────

def spawn_tile(board: np.ndarray, rng: np.random.Generator) -> bool:
    """Place one tile: 90% a "2", 10% a "4". Mutates `board`."""
    free = np.flatnonzero(board == 0)
    if free.size == 0:
        return False
    board[rng.choice(free)] = 1 if rng.random() < 0.9 else 2
    return True


def new_board(rng: np.random.Generator) -> np.ndarray:
    board = np.zeros(16, dtype=np.uint8)
    spawn_tile(board, rng)
    spawn_tile(board, rng)
    return board


# ── Status ───────────────────────────────────────────────────────

def legal_moves(board: np.ndarray) -> list[int]:
    return [d for d in ALL_DIRS if move(board, d)[2]]


def is_game_over(board: np.ndarray) -> bool:
    if (board == 0).any():
        return False
    return not legal_moves(board)


def max_tile(board: np.ndarray) -> int:
    return int(board.max())


def face_value(exponent: int) -> int:
    return 0 if exponent == 0 else 1 << exponent


def render(board: np.ndarray) -> str:
    rows = []
    for r in range(4):
        cells = [face_value(int(board[r * 4 + c])) for c in range(4)]
        rows.append(" ".join(f"{v or '.':>5}" for v in cells))
    return "\n".join(rows)


# ── Symmetry ─────────────────────────────────────────────────────
# The 8 symmetries of the square (D4). Used for n-tuple weight sharing:
# a pattern learned in one corner should be known in all four.

def _build_symmetries() -> list[np.ndarray]:
    identity = np.arange(16, dtype=np.uint8)

    def transpose(cells):
        return cells.reshape(4, 4).T.reshape(16).copy()

    def flip_h(cells):
        return cells.reshape(4, 4)[:, ::-1].reshape(16).copy()

    out = []
    current = identity
    for _ in range(4):
        out.append(current)
        out.append(flip_h(current))
        current = flip_h(transpose(current))  # rotate 90°
    return out


SYMMETRIES = _build_symmetries()

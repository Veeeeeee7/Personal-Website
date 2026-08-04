"""The n-tuple value network, and the config that both the trainer and
the browser agree on.

This file is the source of truth for the network's shape. `export.py`
writes the config into manifest.json verbatim, and src/game/policy.ts
reads it back — so changing a tuple here changes the deployed agent, and
nothing else needs editing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine import SYMMETRIES, WIN_EXPONENT


@dataclass
class NTupleConfig:
    """
    alphabet
        Number of distinct tile symbols the tables index. Because the
        agent stops at 2048, no tile above exponent 11 ever appears, so
        12 symbols covers the whole reachable game. This is not a
        detail — it's the difference between a 4 MB download and a 24 MB
        one, since table size is alphabet ** tuple_length.

    tuples
        Each is a list of board cell indices. Longer tuples see more
        context and cost exponentially more memory.

    symmetric
        Look each tuple up under all 8 board symmetries, summing into one
        shared table. Costs 8x the lookups, saves 8x the samples.
    """

    alphabet: int
    tuples: list[list[int]]
    symmetric: bool = True
    name: str = "custom"

    @property
    def sizes(self) -> list[int]:
        return [self.alphabet ** len(t) for t in self.tuples]

    @property
    def total_weights(self) -> int:
        return sum(self.sizes)

    def describe(self) -> str:
        mb16 = self.total_weights * 2 / 1e6
        return (
            f"{self.name}: {len(self.tuples)} tuples, alphabet {self.alphabet}, "
            f"{self.total_weights:,} weights ({mb16:.1f} MB as int16)"
        )


# Eight 5-tuples: four rows/row-pairs and four 2x2-ish blocks. With
# symmetry these cover every line and every corner of the board.
#
# 8 x 12^5 = ~2.0M weights = 4 MB as int16, ~2 MB over the wire gzipped.
# That is the largest thing worth shipping to a personal site, and it is
# comfortably strong enough to reach 2048 nearly every game.
WEB = NTupleConfig(
    name="web-5tuple",
    alphabet=WIN_EXPONENT + 1,  # 0..11 inclusive
    tuples=[
        [0, 1, 2, 3, 4],
        [4, 5, 6, 7, 8],
        [0, 1, 2, 4, 5],
        [4, 5, 6, 8, 9],
        [0, 1, 4, 5, 8],
        [1, 2, 5, 6, 9],
        [5, 6, 9, 10, 13],
        [1, 2, 5, 6, 10],
    ],
)

# The classic 4x6-tuple network (Wu et al. 2014). Stronger, but 12^6 per
# table puts it at ~24 MB — train with this locally to see the ceiling,
# don't deploy it.
LARGE = NTupleConfig(
    name="large-6tuple",
    alphabet=WIN_EXPONENT + 1,
    tuples=[
        [0, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [0, 1, 2, 4, 5, 6],
        [4, 5, 6, 8, 9, 10],
    ],
)

CONFIGS = {c.name: c for c in (WEB, LARGE)}


class NTupleNetwork:
    """V(s) = sum of one table lookup per (tuple, symmetry) pair."""

    def __init__(self, config: NTupleConfig):
        self.config = config
        self.weights = np.zeros(config.total_weights, dtype=np.float32)

        offsets = np.cumsum([0] + config.sizes[:-1])

        # Flatten every (tuple, symmetry) pair into one lookup plan, so
        # evaluating a board is a single vectorised gather rather than a
        # Python loop over tuples.
        syms = SYMMETRIES if config.symmetric else [SYMMETRIES[0]]
        length = len(config.tuples[0])
        if any(len(t) != length for t in config.tuples):
            # Ragged tuple lengths would break the vectorised gather.
            raise ValueError("all tuples must currently be the same length")

        cells, mults, offs = [], [], []
        for t_index, tuple_cells in enumerate(config.tuples):
            place = np.array(
                [config.alphabet**k for k in range(len(tuple_cells))], dtype=np.int64
            )
            for sym in syms:
                cells.append([sym[c] for c in tuple_cells])
                mults.append(place)
                offs.append(offsets[t_index])

        self.cells = np.array(cells, dtype=np.int64)      # (paths, L)
        self.mults = np.array(mults, dtype=np.int64)      # (paths, L)
        self.offsets = np.array(offs, dtype=np.int64)     # (paths,)
        self.n_paths = self.cells.shape[0]
        self.max_symbol = config.alphabet - 1

    def indices(self, board: np.ndarray) -> np.ndarray:
        """The weight index each lookup path reads for this board."""
        # Clamp rather than error: a tile above the trained alphabet is
        # unreachable in normal play, but a stray one shouldn't crash.
        clamped = np.minimum(board, self.max_symbol).astype(np.int64)
        return (clamped[self.cells] * self.mults).sum(axis=1) + self.offsets

    def value(self, board: np.ndarray) -> float:
        return float(self.weights[self.indices(board)].sum())

    def update(self, board: np.ndarray, delta: float) -> None:
        """Distribute a TD error across the features that produced it.

        Dividing by n_paths keeps the effective step size independent of
        how many tuples and symmetries the config happens to use — so
        alpha means the same thing across configs.
        """
        idx = self.indices(board)
        np.add.at(self.weights, idx, np.float32(delta / self.n_paths))

    # ── Persistence (training checkpoints, not the web format) ────
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            weights=self.weights,
            alphabet=self.config.alphabet,
            tuples=np.array(self.config.tuples),
            symmetric=self.config.symmetric,
            name=self.config.name,
        )

    @classmethod
    def load(cls, path: str) -> "NTupleNetwork":
        data = np.load(path, allow_pickle=False)
        config = NTupleConfig(
            alphabet=int(data["alphabet"]),
            tuples=data["tuples"].tolist(),
            symmetric=bool(data["symmetric"]),
            name=str(data["name"]),
        )
        net = cls(config)
        net.weights = data["weights"].astype(np.float32)
        return net

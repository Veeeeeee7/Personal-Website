// ─────────────────────────────────────────────────────────────
// The stand-in agent: hand-tuned expectimax, no learning involved.
//
// It exists so the page is never broken. Before you publish a trained
// network it plays every game; afterwards it stays as the fallback if
// the weights fail to load. It is deliberately decent-but-beatable —
// it reaches 2048 often, not always, which leaves your network some
// room to look good.
// ─────────────────────────────────────────────────────────────

import {
  type Board,
  type Dir,
  ALL_DIRS,
  CELLS,
  emptyCells,
  faceValue,
  move,
} from './engine';
import type { Decision, Policy } from './policy';

/**
 * Snake weights. Rewards keeping the big tiles pinned along a
 * boustrophedon path from the top-left, which is the standard human
 * strategy and keeps merges available.
 */
// prettier-ignore
const SNAKE = new Float64Array([
  65536, 32768, 16384, 8192,
    512,  1024,  2048, 4096,
    256,   128,    64,   32,
      2,     4,     8,   16,
]);

const EMPTY_BONUS = 4096;
const SMOOTH_PENALTY = 48;

function evaluate(board: Board): number {
  let score = 0;

  for (let i = 0; i < CELLS; i++) {
    if (board[i] !== 0) score += faceValue(board[i]) * SNAKE[i];
  }

  // Open cells are the scarce resource in 2048 — running out is how
  // you lose, long before you run out of merges.
  score += emptyCells(board).length * EMPTY_BONUS;

  // Penalize neighbouring tiles of very different magnitude: they can
  // never merge and they wall off the board.
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const v = board[r * 4 + c];
      if (v === 0) continue;
      if (c < 3 && board[r * 4 + c + 1] !== 0) {
        score -= Math.abs(v - board[r * 4 + c + 1]) * SMOOTH_PENALTY;
      }
      if (r < 3 && board[(r + 1) * 4 + c] !== 0) {
        score -= Math.abs(v - board[(r + 1) * 4 + c]) * SMOOTH_PENALTY;
      }
    }
  }

  return score;
}

const LOSS = -1e9;

function maxNode(board: Board, depth: number): number {
  if (depth === 0) return evaluate(board);

  let best = LOSS;
  for (const dir of ALL_DIRS) {
    const result = move(board, dir);
    if (!result.moved) continue;
    best = Math.max(best, result.reward + chanceNode(result.board, depth - 1));
  }
  return best === LOSS ? LOSS : best;
}

function chanceNode(board: Board, depth: number): number {
  if (depth === 0) return evaluate(board);

  const free = emptyCells(board);
  if (free.length === 0) return maxNode(board, depth);

  // A full expectation over every spawn is exact but quadratic in the
  // open cells. With a crowded board that's cheap; with an open board
  // the position is uncontroversial anyway, so we sample the worst
  // few cells rather than all of them.
  const scan = free.length > 6 ? free.slice(0, 6) : free;

  let total = 0;
  const work = board.slice();
  for (const cell of scan) {
    work[cell] = 1;
    total += 0.9 * maxNode(work, depth - 1);
    work[cell] = 2;
    total += 0.1 * maxNode(work, depth - 1);
    work[cell] = 0;
  }
  return total / scan.length;
}

export class HeuristicPolicy implements Policy {
  readonly name = 'expectimax (heuristic)';

  select(board: Board): Decision | null {
    // Search deeper once the board tightens up — that's where the game
    // is actually decided, and where the branching factor is small
    // enough to afford it.
    const free = emptyCells(board).length;
    const depth = free > 8 ? 2 : free > 4 ? 3 : 4;

    const scores = new Map<Dir, number>();
    let best: Dir | null = null;
    let bestScore = -Infinity;

    for (const dir of ALL_DIRS) {
      const result = move(board, dir);
      if (!result.moved) continue;

      const score = result.reward + chanceNode(result.board, depth - 1);
      scores.set(dir, score);

      if (score > bestScore) {
        bestScore = score;
        best = dir;
      }
    }

    return best === null ? null : { dir: best, scores };
  }
}

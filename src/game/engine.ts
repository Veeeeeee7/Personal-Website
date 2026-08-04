// ─────────────────────────────────────────────────────────────
// Headless 2048 engine. No DOM, no rendering, no randomness that
// isn't injected. The UI and the RL agent both sit on top of this,
// so they are guaranteed to be playing the same game.
//
// Board is a Uint8Array(16) of EXPONENTS, not face values:
//   0 = empty, 1 = tile "2", 2 = "4", ... 11 = "2048".
// Exponents keep the board in one nibble per cell, which is what
// makes the row lookup table below possible.
//
// Cells are row-major:  index = row * 4 + col
//     0  1  2  3
//     4  5  6  7
//     8  9 10 11
//    12 13 14 15
// ─────────────────────────────────────────────────────────────

export type Board = Uint8Array;

export const SIZE = 4;
export const CELLS = 16;

/** Exponent of the winning tile. 2^11 = 2048. */
export const WIN_EXPONENT = 11;

/** Highest exponent the engine will merge into. 2^15 = 32768. */
export const MAX_EXPONENT = 15;

export const Dir = { Up: 0, Right: 1, Down: 2, Left: 3 } as const;
export type Dir = (typeof Dir)[keyof typeof Dir];

export const ALL_DIRS: Dir[] = [Dir.Up, Dir.Right, Dir.Down, Dir.Left];

export const DIR_NAME: Record<Dir, string> = {
  [Dir.Up]: 'up',
  [Dir.Right]: 'right',
  [Dir.Down]: 'down',
  [Dir.Left]: 'left',
};

// ── Cell addressing ───────────────────────────────────────────
// Every direction is expressed as 4 "lines" of 4 positions, where
// position 0 is the edge tiles slide TOWARD. That way a single
// slide-toward-zero routine covers all four moves.
//
//   line l, position p  ->  board cell index
type CellMap = (line: number, pos: number) => number;

const CELL_OF: Record<Dir, CellMap> = {
  [Dir.Left]: (l, p) => l * 4 + p,
  [Dir.Right]: (l, p) => l * 4 + (3 - p),
  [Dir.Up]: (l, p) => p * 4 + l,
  [Dir.Down]: (l, p) => (3 - p) * 4 + l,
};

/** Precomputed cell index for [dir][line][pos]. Avoids a call per lookup. */
const CELL_INDEX: Uint8Array[] = ALL_DIRS.map((dir) => {
  const table = new Uint8Array(16);
  for (let l = 0; l < 4; l++) {
    for (let p = 0; p < 4; p++) table[l * 4 + p] = CELL_OF[dir](l, p);
  }
  return table;
});

// ── Row lookup table ──────────────────────────────────────────
// A line is 4 nibbles packed little-end-first:
//   key = p0 | p1<<4 | p2<<8 | p3<<12
// All 65536 possible lines are precomputed once at module load
// (~2ms). Every move afterwards is four table reads.

interface LineResult {
  /** Packed line after sliding toward position 0. */
  packed: number;
  /** Sum of face values created by merges on this line. */
  reward: number;
  /** dest[p] = final position of the tile that started at p, or -1 if empty. */
  dest: Int8Array;
  /** Bitmask of destination positions that were the product of a merge. */
  mergedMask: number;
  /** True if any tile changed position or merged. */
  moved: boolean;
}

function computeLine(key: number): LineResult {
  const src = [key & 0xf, (key >> 4) & 0xf, (key >> 8) & 0xf, (key >> 12) & 0xf];

  const dest = new Int8Array(4).fill(-1);
  const out: number[] = [];
  let reward = 0;
  let mergedMask = 0;

  // Occupied positions, in slide order.
  const filled: number[] = [];
  for (let p = 0; p < 4; p++) if (src[p] !== 0) filled.push(p);

  let i = 0;
  while (i < filled.length) {
    const p = filled[i];
    const v = src[p];
    const q = i + 1 < filled.length ? filled[i + 1] : -1;

    // Two equal tiles merge — but only one merge per tile per move,
    // which falling through to i += 2 enforces naturally.
    if (q !== -1 && src[q] === v && v < MAX_EXPONENT) {
      const d = out.length;
      dest[p] = d;
      dest[q] = d;
      out.push(v + 1);
      reward += 1 << (v + 1);
      mergedMask |= 1 << d;
      i += 2;
    } else {
      const d = out.length;
      dest[p] = d;
      out.push(v);
      i += 1;
    }
  }

  let packed = 0;
  for (let p = 0; p < out.length; p++) packed |= out[p] << (p * 4);

  return { packed, reward, dest, mergedMask, moved: packed !== key };
}

const LINE_TABLE: LineResult[] = (() => {
  const table = new Array<LineResult>(65536);
  for (let key = 0; key < 65536; key++) table[key] = computeLine(key);
  return table;
})();

// ── Move ──────────────────────────────────────────────────────

/** One tile's journey during a move. Both halves of a merge get an entry. */
export interface Slide {
  from: number;
  to: number;
}

export interface MoveResult {
  /** The afterstate: board post-slide, PRE-spawn. This is what the agent scores. */
  board: Board;
  /** Score gained, i.e. the sum of merged tile face values. */
  reward: number;
  /** False if the move was illegal (nothing shifted). Board is then unchanged. */
  moved: boolean;
  /** Tile movements, for animation. Empty when !moved. */
  slides: Slide[];
  /** Cells that received a merge, for the pop animation. Empty when !moved. */
  merges: number[];
}

export function emptyBoard(): Board {
  return new Uint8Array(CELLS);
}

export function cloneBoard(board: Board): Board {
  return board.slice();
}

/**
 * Apply a move. Pure: `board` is never mutated.
 *
 * The returned board is the AFTERSTATE — no new tile has spawned yet.
 * That separation matters: afterstate TD learning scores exactly this
 * position, so the agent and the trainer must both see it.
 */
export function move(board: Board, dir: Dir): MoveResult {
  const cells = CELL_INDEX[dir];
  const next = emptyBoard();
  const slides: Slide[] = [];
  const merges: number[] = [];
  let reward = 0;
  let moved = false;

  for (let l = 0; l < 4; l++) {
    const base = l * 4;

    const key =
      board[cells[base]] |
      (board[cells[base + 1]] << 4) |
      (board[cells[base + 2]] << 8) |
      (board[cells[base + 3]] << 12);

    const result = LINE_TABLE[key];
    if (result.moved) moved = true;
    reward += result.reward;

    // Unpack the resulting line back into board cells.
    for (let p = 0; p < 4; p++) {
      next[cells[base + p]] = (result.packed >> (p * 4)) & 0xf;
    }

    // Translate line-local traces into board cell indices.
    for (let p = 0; p < 4; p++) {
      const d = result.dest[p];
      if (d === -1) continue;
      slides.push({ from: cells[base + p], to: cells[base + d] });
    }
    for (let d = 0; d < 4; d++) {
      if (result.mergedMask & (1 << d)) merges.push(cells[base + d]);
    }
  }

  if (!moved) {
    return { board: cloneBoard(board), reward: 0, moved: false, slides: [], merges: [] };
  }

  return { board: next, reward, moved: true, slides, merges };
}

// ── Spawning ──────────────────────────────────────────────────

/** Injected so the trainer, the tests, and the page can all be deterministic. */
export type Rng = () => number;

export interface Spawn {
  cell: number;
  value: number;
}

export function emptyCells(board: Board): number[] {
  const out: number[] = [];
  for (let i = 0; i < CELLS; i++) if (board[i] === 0) out.push(i);
  return out;
}

/**
 * Place one random tile: 90% a "2", 10% a "4" — the standard 2048
 * distribution. Mutates `board` and returns what it placed, or null
 * if there was no room.
 */
export function spawnTile(board: Board, rng: Rng = Math.random): Spawn | null {
  const free = emptyCells(board);
  if (free.length === 0) return null;

  const cell = free[Math.floor(rng() * free.length)];
  const value = rng() < 0.9 ? 1 : 2;
  board[cell] = value;
  return { cell, value };
}

/** A fresh game: empty board plus the two starting tiles. */
export function newBoard(rng: Rng = Math.random): { board: Board; spawns: Spawn[] } {
  const board = emptyBoard();
  const spawns: Spawn[] = [];
  for (let i = 0; i < 2; i++) {
    const s = spawnTile(board, rng);
    if (s) spawns.push(s);
  }
  return { board, spawns };
}

// ── Status ────────────────────────────────────────────────────

export function legalMoves(board: Board): Dir[] {
  return ALL_DIRS.filter((dir) => move(board, dir).moved);
}

export function isGameOver(board: Board): boolean {
  if (emptyCells(board).length > 0) return false;
  return legalMoves(board).length === 0;
}

export function maxTile(board: Board): number {
  let max = 0;
  for (let i = 0; i < CELLS; i++) if (board[i] > max) max = board[i];
  return max;
}

export function hasWon(board: Board, target = WIN_EXPONENT): boolean {
  return maxTile(board) >= target;
}

/** Face value of an exponent. 0 stays 0 (empty), otherwise 2^e. */
export function faceValue(exponent: number): number {
  return exponent === 0 ? 0 : 1 << exponent;
}

// ── Symmetry ──────────────────────────────────────────────────
// The 8 symmetries of the square (dihedral group D4). The n-tuple
// network uses these for weight sharing: a pattern in one corner
// should be worth what the same pattern is worth in every other
// corner, so each tuple is looked up under all 8 transforms.

function transpose(cells: Uint8Array): Uint8Array {
  const out = new Uint8Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[c * 4 + r] = cells[r * 4 + c];
  return out;
}

function flipHorizontal(cells: Uint8Array): Uint8Array {
  const out = new Uint8Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[r * 4 + (3 - c)] = cells[r * 4 + c];
  return out;
}

/**
 * SYMMETRIES[s] maps a position to the cell it reads from under
 * transform s. Applied to tuple cell lists, not to boards — that
 * lets the policy precompute 8 index paths per tuple and then do
 * nothing but adds and lookups at inference time.
 */
export const SYMMETRIES: Uint8Array[] = (() => {
  const identity = new Uint8Array(16);
  for (let i = 0; i < 16; i++) identity[i] = i;

  const out: Uint8Array[] = [];
  let current = identity;

  // 4 rotations, each with and without a mirror.
  for (let r = 0; r < 4; r++) {
    out.push(current);
    out.push(flipHorizontal(current));
    current = flipHorizontal(transpose(current)); // rotate 90°
  }
  return out;
})();

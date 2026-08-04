// ─────────────────────────────────────────────────────────────
// The policy layer: everything that decides which move to play.
//
// `Policy` is the seam between the website and your RL work. The UI
// knows nothing except this interface, so a trained network and the
// built-in heuristic are interchangeable — swap one for the other
// without touching a line of the page.
// ─────────────────────────────────────────────────────────────

import {
  type Board,
  type Dir,
  ALL_DIRS,
  CELLS,
  SYMMETRIES,
  move,
} from './engine';

/** What the agent decided, and how it felt about the alternatives. */
export interface Decision {
  dir: Dir;
  /** Value estimate per legal direction. Drives the stats overlay. */
  scores: Map<Dir, number>;
}

export interface Policy {
  readonly name: string;
  /** Null means no legal move exists — the game is over. */
  select(board: Board): Decision | null;
}

// ── Weight file format ────────────────────────────────────────
// Two files live in public/policy/:
//
//   manifest.json   the shapes and the quantization scale
//   weights.bin     raw little-endian weights, tables concatenated
//                   in the same order as `tuples`
//
// Table t holds alphabet ** tuples[t].length entries. Splitting the
// metadata from the payload means the .bin stays byte-aligned and
// streams straight into a typed array with no parsing.

export interface PolicyManifest {
  format: 'ntuple-v1';
  /** Number of distinct tile symbols. Exponents >= this are clamped. */
  alphabet: number;
  /** Each tuple is a list of board cell indices (0..15). */
  tuples: number[][];
  /** Look each tuple up under all 8 board symmetries (weight sharing). */
  symmetric: boolean;
  quant: {
    dtype: 'int16' | 'int32' | 'float32';
    /** Real weight = stored * scale. Use 1 for float32. */
    scale: number;
  };
  /** Weights filename, resolved relative to the manifest. */
  weights: string;
  /** Free-form; surfaced in the UI. */
  meta?: {
    trainedEpisodes?: number;
    meanScore?: number;
    winRate?: number;
    note?: string;
  };
}

// ── N-tuple network ───────────────────────────────────────────

/**
 * A linear value function over tile patterns — the approach that
 * actually solves 2048 (Szubert & Jaśkowski 2014). V(s) is a sum of
 * table lookups, so inference is a few dozen array reads: fast
 * enough to run thousands of moves per second in a browser tab.
 */
export class NTuplePolicy implements Policy {
  readonly name: string;

  private readonly alphabet: number;
  private readonly scale: number;
  private readonly weights: Int16Array | Int32Array | Float32Array;

  /**
   * Flattened lookup plan. For every (tuple, symmetry) pair we store
   * the 8 cell indices to read and the place-value multiplier for
   * each, already offset into the global weight array. Inference then
   * needs no per-move branching.
   */
  private readonly paths: Array<{ cells: Uint8Array; mults: Int32Array; offset: number }>;

  constructor(manifest: PolicyManifest, buffer: ArrayBuffer) {
    this.name = 'n-tuple network';
    this.alphabet = manifest.alphabet;
    this.scale = manifest.quant.scale;

    this.weights =
      manifest.quant.dtype === 'int16'
        ? new Int16Array(buffer)
        : manifest.quant.dtype === 'int32'
          ? new Int32Array(buffer)
          : new Float32Array(buffer);

    // Table offsets: tables are concatenated in manifest order.
    const offsets: number[] = [];
    let cursor = 0;
    for (const tuple of manifest.tuples) {
      offsets.push(cursor);
      cursor += this.alphabet ** tuple.length;
    }

    if (cursor !== this.weights.length) {
      throw new Error(
        `policy: manifest expects ${cursor} weights but weights.bin holds ${this.weights.length}`,
      );
    }

    const symmetries = manifest.symmetric ? SYMMETRIES : [SYMMETRIES[0]];

    this.paths = [];
    for (let t = 0; t < manifest.tuples.length; t++) {
      const tuple = manifest.tuples[t];

      const mults = new Int32Array(tuple.length);
      for (let k = 0; k < tuple.length; k++) mults[k] = this.alphabet ** k;

      for (const sym of symmetries) {
        const cells = new Uint8Array(tuple.length);
        for (let k = 0; k < tuple.length; k++) cells[k] = sym[tuple[k]];
        this.paths.push({ cells, mults, offset: offsets[t] });
      }
    }
  }

  /** V(board). The board passed here is an afterstate, pre-spawn. */
  value(board: Board): number {
    // Tiles above the trained alphabet are clamped rather than thrown
    // out — the network has never seen them, but a clamped lookup
    // degrades gracefully where an out-of-bounds read would not.
    const max = this.alphabet - 1;
    let sum = 0;

    for (let i = 0; i < this.paths.length; i++) {
      const { cells, mults, offset } = this.paths[i];
      let index = 0;
      for (let k = 0; k < cells.length; k++) {
        const v = board[cells[k]];
        index += (v > max ? max : v) * mults[k];
      }
      sum += this.weights[offset + index];
    }

    return sum * this.scale;
  }

  select(board: Board): Decision | null {
    const scores = new Map<Dir, number>();
    let best: Dir | null = null;
    let bestScore = -Infinity;

    for (const dir of ALL_DIRS) {
      const result = move(board, dir);
      if (!result.moved) continue;

      // Afterstate evaluation: immediate reward plus the learned value
      // of the position before the random tile lands. No expectation
      // over spawns needed — that's the whole point of afterstates.
      const score = result.reward + this.value(result.board);
      scores.set(dir, score);

      if (score > bestScore) {
        bestScore = score;
        best = dir;
      }
    }

    return best === null ? null : { dir: best, scores };
  }
}

// ── Loading ───────────────────────────────────────────────────

export const POLICY_URL = '/policy/manifest.json';

/**
 * Fetch the trained network. Resolves to null when no policy has been
 * published yet — the page treats that as "fall back to the heuristic"
 * rather than an error, so the site works before the model exists.
 */
export async function loadNTuplePolicy(
  url = POLICY_URL,
): Promise<{ policy: NTuplePolicy; manifest: PolicyManifest } | null> {
  try {
    const manifestResponse = await fetch(url, { cache: 'force-cache' });
    if (!manifestResponse.ok) return null;

    const manifest = (await manifestResponse.json()) as PolicyManifest;
    if (manifest.format !== 'ntuple-v1') {
      console.warn(`policy: unknown format "${manifest.format}"`);
      return null;
    }

    const weightsUrl = new URL(manifest.weights, new URL(url, location.href)).href;
    const weightsResponse = await fetch(weightsUrl, { cache: 'force-cache' });
    if (!weightsResponse.ok) return null;

    const buffer = await weightsResponse.arrayBuffer();
    return { policy: new NTuplePolicy(manifest, buffer), manifest };
  } catch (error) {
    console.warn('policy: falling back to heuristic —', error);
    return null;
  }
}

// ── Sanity check ──────────────────────────────────────────────

/** Cheap guard against a manifest that would allocate a gigabyte. */
export function manifestSize(manifest: PolicyManifest): number {
  return manifest.tuples.reduce((sum, t) => sum + manifest.alphabet ** t.length, 0);
}

export function assertBoardShape(board: Board): void {
  if (board.length !== CELLS) throw new Error(`policy: expected ${CELLS} cells`);
}

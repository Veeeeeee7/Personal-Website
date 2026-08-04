// ─────────────────────────────────────────────────────────────
// The transport: play / pause / speed for an autoplaying policy.
//
// Deliberately knows nothing about 2048 or about the DOM. It is given
// a `step` function, and its whole job is deciding how often to call
// it and yielding to the browser in between so the tab stays alive
// even at thousands of moves per second.
// ─────────────────────────────────────────────────────────────

/** What a single step reported back. Anything but 'ok' halts the loop. */
export type StepResult = 'ok' | 'won' | 'gameover' | 'stuck';

export interface SpeedPreset {
  label: string;
  /** Target moves per second. Infinity = as fast as the frame allows. */
  mps: number;
}

/**
 * 1× (the default) is a calm ~2 moves/sec — slow enough to actually
 * follow what the agent is doing. Each label's rate is deliberately
 * gentle; the top of the range exists because watching a solved game at
 * that pace gets boring around move 300, and a full run is ~1200 moves.
 */
export const SPEEDS: SpeedPreset[] = [
  { label: '0.5×', mps: 1 },
  { label: '1×', mps: 2 },
  { label: '2×', mps: 4 },
  { label: '5×', mps: 10 },
  { label: 'Max', mps: Infinity },
];

/** Index of "1×" — a watchable default, not a blur. */
export const DEFAULT_SPEED_INDEX = 1;

/** Milliseconds per frame we're willing to spend stepping at Max. */
const FRAME_BUDGET_MS = 8;

/** Hard cap so a pathological policy can't wedge the tab. */
const MAX_STEPS_PER_FRAME = 2000;

export interface AgentRunnerOptions {
  /** Performs exactly one move. Must be synchronous. */
  step: () => StepResult;
  /** Fired when the loop halts on its own (win, loss, or no legal move). */
  onHalt?: (reason: StepResult) => void;
  /** Fired whenever running state flips, so the UI can re-render controls. */
  onStateChange?: (running: boolean) => void;
}

export class AgentRunner {
  private raf: number | null = null;
  private accumulator = 0;
  private lastTimestamp = 0;
  private speedIndex = DEFAULT_SPEED_INDEX;

  constructor(private readonly options: AgentRunnerOptions) {}

  get running(): boolean {
    return this.raf !== null;
  }

  get speed(): SpeedPreset {
    return SPEEDS[this.speedIndex];
  }

  get speedIndex_(): number {
    return this.speedIndex;
  }

  setSpeedIndex(index: number): void {
    this.speedIndex = Math.max(0, Math.min(SPEEDS.length - 1, index));
    // Drop any banked time: a speed change should take effect now, not
    // fire a burst of catch-up moves from the old interval.
    this.accumulator = 0;
  }

  start(): void {
    if (this.running) return;
    this.lastTimestamp = performance.now();
    this.accumulator = 0;
    this.raf = requestAnimationFrame(this.tick);
    this.options.onStateChange?.(true);
  }

  pause(): void {
    if (this.raf === null) return;
    cancelAnimationFrame(this.raf);
    this.raf = null;
    this.options.onStateChange?.(false);
  }

  toggle(): void {
    this.running ? this.pause() : this.start();
  }

  private halt(reason: StepResult): void {
    this.pause();
    this.options.onHalt?.(reason);
  }

  private tick = (timestamp: number): void => {
    // Re-arm first: any `return` below still leaves the loop alive
    // unless something explicitly halted it.
    this.raf = requestAnimationFrame(this.tick);

    const elapsed = Math.min(timestamp - this.lastTimestamp, 250); // clamp tab-restore jumps
    this.lastTimestamp = timestamp;

    if (this.speed.mps === Infinity) {
      const deadline = performance.now() + FRAME_BUDGET_MS;
      for (let i = 0; i < MAX_STEPS_PER_FRAME; i++) {
        const result = this.options.step();
        if (result !== 'ok') return this.halt(result);
        if (performance.now() >= deadline) break;
      }
      return;
    }

    const interval = 1000 / this.speed.mps;
    this.accumulator += elapsed;

    let guard = 0;
    while (this.accumulator >= interval && guard++ < MAX_STEPS_PER_FRAME) {
      this.accumulator -= interval;
      const result = this.options.step();
      if (result !== 'ok') return this.halt(result);
    }
  };

  destroy(): void {
    if (this.raf !== null) cancelAnimationFrame(this.raf);
    this.raf = null;
  }
}

/**
 * How long tiles should take to slide, given the current move rate.
 * The animation has to finish before the next move starts or tiles
 * visibly teleport mid-transition; past ~20 moves/sec there's no point
 * animating at all, so we cut it to zero and let the board snap.
 */
export function animationDuration(mps: number): number {
  if (mps === Infinity || mps > 25) return 0;
  return Math.min(110, (1000 / mps) * 0.55);
}

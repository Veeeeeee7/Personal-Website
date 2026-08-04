# 2048 — reinforcement learning

Trains an n-tuple network to play 2048, and exports it to a form the
website can run in a browser tab with no ML runtime.

```
train.py    TD(0) learning over afterstates
ntuple.py   the value network + the tuple configs
engine.py   game rules (mirror of ../src/game/engine.ts)
export.py   checkpoint -> ../public/policy/{manifest.json,weights.bin}
verify.py   generates cross-check fixtures for the TS side
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy numba      # numba is optional but ~50x faster
```

## Train

```bash
cd rl
python train.py --episodes 200000
```

Expect the mean score to climb for a long time. Rough milestones with
the default `web-5tuple` config:

| episodes | mean score | 2048 rate |
| -------- | ---------- | --------- |
| 1k       | ~8,000     | a few %   |
| 20k      | ~20,000    | ~40%      |
| 100k     | ~35,000    | ~85%      |
| 300k+    | ~45,000    | ~95%+     |

Checkpoints land in `checkpoints/` every 10k episodes and are resumable:

```bash
python train.py --resume checkpoints/web-5tuple.npz --episodes 100000
```

If the score plateaus, drop `--alpha` (0.1 → 0.05 → 0.025). Annealing the
learning rate is usually worth more than more episodes at a fixed one.

## Export

```bash
python export.py --checkpoint checkpoints/web-5tuple.npz --episodes 200000
```

Writes `../public/policy/`. The page picks it up on the next load and
swaps out the fallback heuristic automatically — no code change. Commit
those two files; they are the deployed agent.

## Verify

Two seams can break silently, so both are checked explicitly.

```bash
# Do engine.py and engine.ts agree on the rules?
python verify.py && npm run verify:engine

# Does the browser read the exported weights the way Python wrote them?
python verify.py --policy && npm run verify:policy
```

Run these after touching either engine, the tuple config, or the export
format. Both failure modes are nasty precisely because they don't throw:
a rules mismatch means the agent is evaluated on a game it wasn't
trained on, and a weight-layout mismatch produces an agent that loads
fine and plays like it never learned. The engine check has already
caught one real bug (a `uint8 << 12` overflow in the Python row packer).

Needs Node 22.6+ — the scripts import the `.ts` sources directly using
Node's built-in type stripping, so there's no build step and no risk of
verifying a stale copy.

## Design notes

**Why n-tuples and not a neural network.** 2048 punishes the obvious
approaches. Search burns its budget on futures the random tile spawn
never realizes; deep RL faces a sparse reward, a huge state space, and
has to relearn each pattern in all four corners. An n-tuple network is a
linear value function over small cell patterns — V(s) is a sum of table
lookups. It trains in hours on a CPU, beats every deep RL result on this
game, and deploys as an array.

**Why afterstates.** An afterstate is the board after your slide and
merge but *before* the random tile lands. It's a deterministic
consequence of your move, so a value function defined on afterstates
never averages over spawn positions:

```
V(s'_t) <- V(s'_t) + alpha * [ r_{t+1} + V(s'_{t+1}) - V(s'_t) ]
```

Move selection is then just `argmax over dirs of (reward + V(afterstate))`.
No expectimax, no rollouts.

**Why alphabet 12.** Episodes stop at the 2048 tile, matching what the
website does, so no tile above exponent 11 ever appears. Table size is
`alphabet ** tuple_length`, so this choice is what keeps the download at
4 MB instead of 24 MB. Training past 2048 needs a larger alphabet — and
a different plan for shipping it.

**Why 5-tuples.** `8 x 12^5` ≈ 2.0M weights = 4 MB as int16 (~2 MB
gzipped). The classic 4×6-tuple network (`--config large`) is stronger
but lands at 24 MB, which is not a reasonable thing to put on a personal
site. Train it locally if you want to see the ceiling.

// Check that the browser reads an exported policy exactly as Python wrote it.
//
// verify_engine.mjs proves the two engines agree on the rules. This
// proves the two sides agree on the *weights*: same tuple order, same
// symmetry order, same place-value packing, same quantization scale. A
// silent disagreement here doesn't crash — it just produces an agent
// that plays like it was never trained, which is a miserable thing to
// debug from the outside.
//
// Run `python verify.py --policy` first to write policy_fixtures.json.
//
//   node --experimental-strip-types verify_policy.mjs
//   npm run verify:policy

import { readFileSync } from 'node:fs';
import { register } from 'node:module';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

register('./ts-hooks.mjs', import.meta.url);

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const fixturesPath = join(here, 'policy_fixtures.json');
let fixtures;
try {
  fixtures = JSON.parse(readFileSync(fixturesPath, 'utf8'));
} catch {
  console.error('policy_fixtures.json missing — run `python verify.py --policy` first.');
  process.exit(1);
}

const { NTuplePolicy } = await import(
  pathToFileURL(join(root, 'src', 'game', 'policy.ts')).href
);

const weights = readFileSync(join(root, fixtures.weightsPath));
const buffer = weights.buffer.slice(
  weights.byteOffset,
  weights.byteOffset + weights.byteLength,
);

const policy = new NTuplePolicy(fixtures.manifest, buffer);

// Values run to the thousands, so an absolute epsilon this tight is a
// strict test: it only passes if both sides sum the same integers.
const EPSILON = 1e-6;

let worst = 0;
let mismatches = 0;

for (const testCase of fixtures.cases) {
  const value = policy.value(Uint8Array.from(testCase.board));
  const drift = Math.abs(value - testCase.value);
  if (drift > worst) worst = drift;
  if (drift > EPSILON) {
    if (mismatches < 5) {
      console.error(`FAIL [${testCase.board}]\n     py ${testCase.value}  ts ${value}`);
    }
    mismatches++;
  }
}

// The move choice is what actually ships. Values could match while an
// argmax tie broke the other way, so check the decision too.
let moveMismatches = 0;
for (const testCase of fixtures.cases) {
  if (testCase.bestDir === null) continue;
  const decision = policy.select(Uint8Array.from(testCase.board));
  const dir = decision === null ? null : decision.dir;
  if (dir !== testCase.bestDir) {
    if (moveMismatches < 5) {
      console.error(`FAIL move [${testCase.board}]: py ${testCase.bestDir} ts ${dir}`);
    }
    moveMismatches++;
  }
}

if (mismatches || moveMismatches) {
  console.error(
    `\n${mismatches} value mismatch(es), ${moveMismatches} move mismatch(es) ` +
      `across ${fixtures.cases.length} boards.`,
  );
  process.exit(1);
}

console.log(
  `policy.ts matches ntuple.py on ${fixtures.cases.length} boards ` +
    `(max drift ${worst.toExponential(2)}).`,
);
console.log(`Move choice agrees on every board.`);

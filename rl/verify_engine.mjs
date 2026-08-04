// Replay Python's fixtures against the TypeScript engine.
//
// Run `python verify.py` first to generate fixtures.json, then:
//   node --experimental-strip-types verify_engine.mjs
//
// Or just `npm run verify:engine`, which sets the flag for you.
//
// Node strips the types itself (22.6+), so this needs no build step and
// no toolchain — it imports the exact file the browser bundles.

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const fixturesPath = join(here, 'fixtures.json');
let fixtures;
try {
  fixtures = JSON.parse(readFileSync(fixturesPath, 'utf8'));
} catch {
  console.error('fixtures.json missing — run `python verify.py` first.');
  process.exit(1);
}

let engine;
try {
  engine = await import(join(root, 'src', 'game', 'engine.ts'));
} catch (error) {
  console.error(
    'Could not import engine.ts. This needs Node 22.6+ with --experimental-strip-types:\n' +
      '  node --experimental-strip-types verify_engine.mjs\n',
  );
  throw error;
}

let failures = 0;
const report = (label, detail) => {
  if (failures++ < 10) console.error(`FAIL ${label}\n     ${detail}`);
};

// ── Symmetries ────────────────────────────────────────────────
// Order matters as much as content: the weight tables are indexed by
// symmetry position, so a permuted list silently corrupts every lookup.
fixtures.symmetries.forEach((expected, i) => {
  const actual = Array.from(engine.SYMMETRIES[i] ?? []);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    report(`symmetry[${i}]`, `py ${expected}\n     ts ${actual}`);
  }
});

if (engine.SYMMETRIES.length !== fixtures.symmetries.length) {
  report('symmetry count', `py ${fixtures.symmetries.length} vs ts ${engine.SYMMETRIES.length}`);
}

// ── Moves ─────────────────────────────────────────────────────
let checked = 0;
for (const testCase of fixtures.cases) {
  const board = Uint8Array.from(testCase.board);
  const result = engine.move(board, testCase.dir);

  const after = Array.from(result.board);
  const dirName = engine.DIR_NAME[testCase.dir];

  if (result.moved !== testCase.moved) {
    report(`moved [${testCase.board}] ${dirName}`, `py ${testCase.moved} vs ts ${result.moved}`);
  } else if (result.reward !== testCase.reward) {
    report(`reward [${testCase.board}] ${dirName}`, `py ${testCase.reward} vs ts ${result.reward}`);
  } else if (JSON.stringify(after) !== JSON.stringify(testCase.after)) {
    report(`board [${testCase.board}] ${dirName}`, `py ${testCase.after}\n     ts ${after}`);
  }

  // The engine promises not to mutate its input. The trainer relies on
  // that, and a violation here would be near-impossible to trace later.
  if (JSON.stringify(Array.from(board)) !== JSON.stringify(testCase.board)) {
    report(`mutation [${testCase.board}] ${dirName}`, 'move() mutated its input board');
  }

  // Slides must account for exactly the tiles present, and merges must
  // land where two tiles arrived. Python doesn't model these, so they're
  // checked for internal consistency instead.
  if (result.moved) {
    const tilesBefore = testCase.board.filter((v) => v !== 0).length;
    if (result.slides.length !== tilesBefore) {
      report(`slides [${testCase.board}] ${dirName}`, `${result.slides.length} slides for ${tilesBefore} tiles`);
    }

    const arrivals = new Map();
    for (const s of result.slides) arrivals.set(s.to, (arrivals.get(s.to) ?? 0) + 1);

    const doubled = [...arrivals].filter(([, n]) => n === 2).map(([cell]) => cell).sort();
    const merges = [...result.merges].sort();
    if (JSON.stringify(doubled) !== JSON.stringify(merges)) {
      report(`merges [${testCase.board}] ${dirName}`, `doubled arrivals ${doubled} vs merges ${merges}`);
    }
    if ([...arrivals.values()].some((n) => n > 2)) {
      report(`merges [${testCase.board}] ${dirName}`, 'three tiles arrived in one cell');
    }
  }

  checked++;
}

if (failures) {
  console.error(`\n${failures} failure(s) across ${checked.toLocaleString()} cases.`);
  process.exit(1);
}

console.log(`engine.ts matches engine.py on ${checked.toLocaleString()} move cases.`);
console.log(`${fixtures.symmetries.length} symmetries match, in order.`);

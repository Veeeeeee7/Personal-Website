// Lets Node import the game sources exactly as Vite bundles them.
//
// Vite resolves extensionless relative imports ('./engine'); Node's ESM
// resolver does not. Rather than litter the source with '.ts' suffixes
// to appease a script that only runs in CI, the gap is patched here.
//
// Registered by verify_policy.mjs via module.register().

import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const HAS_EXTENSION = /\.[cm]?[jt]sx?$|\.json$/;

export async function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith('.') && !HAS_EXTENSION.test(specifier)) {
    for (const ext of ['.ts', '.js']) {
      try {
        const candidate = await nextResolve(specifier + ext, context);
        if (existsSync(fileURLToPath(candidate.url))) return candidate;
      } catch {
        // fall through to the next extension
      }
    }
  }
  return nextResolve(specifier, context);
}

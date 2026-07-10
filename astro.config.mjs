import { defineConfig } from 'astro/config';

import cloudflare from "@astrojs/cloudflare";

export default defineConfig({
  site: 'https://victor-li.me',

  // Trailing slashes off keeps /activity clean rather than /activity/
  trailingSlash: 'never',

  build: {
    format: 'file',
  },

  output: "hybrid",
  adapter: cloudflare()
});
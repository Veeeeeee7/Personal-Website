import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://victor-li.me',
  // Trailing slashes off keeps /activity clean rather than /activity/
  trailingSlash: 'never',
  build: {
    format: 'file',
  },
});

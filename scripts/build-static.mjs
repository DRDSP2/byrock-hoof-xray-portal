import { cp, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const files = ['index.html', 'frontend'];
await rm('dist', { recursive: true, force: true });
await mkdir('dist', { recursive: true });
for (const file of files) {
  if (existsSync(file)) await cp(file, `dist/${file}`, { recursive: true });
}
console.log('Static build written to dist/');

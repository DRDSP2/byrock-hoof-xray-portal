import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { existsSync } from 'node:fs';

const execFileAsync = promisify(execFile);

async function hasCommand(command) {
  try {
    await execFileAsync('which', [command]);
    return true;
  } catch {
    return false;
  }
}

if (!existsSync('dist/index.html')) {
  console.error('Missing dist build. Run npm run build first.');
  process.exit(1);
}

if (!(await hasCommand('ipfs'))) {
  console.error('IPFS CLI is not installed or not on PATH. Install/configure ipfs, or set up the project pinning service credentials, then rerun npm run deploy:ipfs.');
  process.exit(2);
}

try {
  const { stdout } = await execFileAsync('ipfs', ['add', '-Qr', 'dist']);
  const cid = stdout.trim().split('\n').filter(Boolean).at(-1);
  console.log(`IPFS CID: ${cid}`);
  console.log(`Gateway URL: https://ipfs.io/ipfs/${cid}/`);
} catch (error) {
  console.error(error.stderr || error.message);
  process.exit(error.code || 1);
}

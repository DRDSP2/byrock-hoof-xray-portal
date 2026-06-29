import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { calculateRotationFromLandmarks, getThresholds } from '../image-processor.js';

const baseLandmarks = [
  { name: 'coronary_band', x: 40, y: 20 },
  { name: 'toe_tip', x: 40, y: 80 },
  { name: 'extensor_process', x: 55, y: 25 },
  { name: 'p3_tip', x: 55, y: 80 },
  { name: 'p3_heel', x: 35, y: 70 },
  { name: 'toe_ground', x: 70, y: 90 },
  { name: 'heel_ground', x: 25, y: 90 }
];

const normal = calculateRotationFromLandmarks(baseLandmarks, { view: 'DP' });
assert.equal(normal.critical, false);
assert.ok(normal.rotation < 5);

const critical = calculateRotationFromLandmarks([
  ...baseLandmarks.filter(l => l.name !== 'p3_tip'),
  { name: 'p3_tip', x: 62, y: 80 }
], { view: 'DP' });
assert.equal(critical.critical, true);
assert.ok(critical.rotation > 5);

const draftThresholds = getThresholds({ breed: 'Clydesdale draft', view: 'DP' });
assert.equal(draftThresholds.active, 7);

const missing = calculateRotationFromLandmarks([{ name: 'toe_tip', x: 1, y: 1 }]);
assert.equal(missing.status, 'needs_landmarks');
assert.ok(missing.missing.includes('coronary_band'));

const frontendHtml = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
assert.match(frontendHtml, /const API_BASE = \(\(\) => \{/);
assert.match(frontendHtml, /api\('\/api\/analyze'/);
assert.doesNotMatch(frontendHtml, /const API_BASE = 'http:\/\/localhost:8000'/);
assert.match(frontendHtml, /function asArray\(value\)/);
assert.match(frontendHtml, /currentScans = asArray\(data\);/);
assert.match(frontendHtml, /currentLandmarks = asArray\(data\)\.map/);
assert.match(frontendHtml, /No scan available for this hoof/);

const nginxConfig = readFileSync(new URL('../../nginx/nginx.conf', import.meta.url), 'utf8');
assert.match(nginxConfig, /location = \/compute/);
assert.match(nginxConfig, /proxy_pass http:\/\/byrock-backend:8000\/api\/analyze;/);

const nginxProdConfig = readFileSync(new URL('../../nginx/nginx.prod.conf', import.meta.url), 'utf8');
assert.match(nginxProdConfig, /location = \/compute/);
assert.match(nginxProdConfig, /proxy_pass http:\/\/byrock-backend:8000\/api\/analyze;/);

console.log('rotation pipeline tests passed');

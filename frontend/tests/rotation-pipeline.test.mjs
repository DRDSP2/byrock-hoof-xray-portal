import assert from 'node:assert/strict';
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

console.log('rotation pipeline tests passed');

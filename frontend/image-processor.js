const DEFAULT_THRESHOLDS = Object.freeze({
  coronal: 5,
  sagittal: 3,
  draftCoronal: 7,
  draftSagittal: 5
});

const LANDMARK_LABELS = Object.freeze({
  coronary_band: 'Coronary Band',
  toe_tip: 'Toe Tip',
  extensor_process: 'Extensor Process',
  p3_tip: 'P3 Tip',
  p3_heel: 'P3 Heel',
  toe_ground: 'Toe Ground',
  heel_ground: 'Heel Ground'
});

function normalizeAngle(angle) {
  if (!Number.isFinite(angle)) return 0;
  let normalized = angle % 360;
  if (normalized < 0) normalized += 360;
  return normalized;
}

function smallestAngleBetween(a, b) {
  const diff = Math.abs(normalizeAngle(a) - normalizeAngle(b)) % 360;
  return diff > 180 ? 360 - diff : diff;
}

function angleBetweenPoints(start, end) {
  return normalizeAngle(Math.atan2(end.y - start.y, end.x - start.x) * 180 / Math.PI);
}

function vectorLength(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function getThresholds({ breed = '', view = 'Lateral', overrides = {} } = {}) {
  const isDraft = /draft|clydesdale|shire|percheron|belgian/i.test(breed || '');
  const coronal = Number(overrides.coronal ?? (isDraft ? DEFAULT_THRESHOLDS.draftCoronal : DEFAULT_THRESHOLDS.coronal));
  const sagittal = Number(overrides.sagittal ?? (isDraft ? DEFAULT_THRESHOLDS.draftSagittal : DEFAULT_THRESHOLDS.sagittal));
  const active = /dp|dorso|palmar|coronal|frontal/i.test(view || '') ? coronal : sagittal;
  return { coronal, sagittal, active, view: /dp|dorso|palmar|coronal|frontal/i.test(view || '') ? 'coronal' : 'sagittal' };
}

function landmarksToMap(landmarks = []) {
  return landmarks.reduce((acc, landmark) => {
    const name = landmark.name || landmark.id;
    if (name && Number.isFinite(Number(landmark.x)) && Number.isFinite(Number(landmark.y))) {
      acc[name] = { x: Number(landmark.x), y: Number(landmark.y), name };
    }
    return acc;
  }, {});
}

function calculateRotationFromLandmarks(landmarks = [], options = {}) {
  const map = Array.isArray(landmarks) ? landmarksToMap(landmarks) : landmarks;
  const required = ['coronary_band', 'toe_tip', 'extensor_process', 'p3_tip'];
  const missing = required.filter(name => !map[name]);
  if (missing.length) {
    return {
      rotation: 0,
      rawAngle: 0,
      critical: false,
      confidence: 0,
      status: 'needs_landmarks',
      missing,
      message: `Place ${missing.map(name => LANDMARK_LABELS[name] || name).join(', ')} to calculate rotation.`
    };
  }

  const wallAngle = angleBetweenPoints(map.coronary_band, map.toe_tip);
  const p3Angle = angleBetweenPoints(map.extensor_process, map.p3_tip);
  const rotation = smallestAngleBetween(wallAngle, p3Angle);
  const thresholds = getThresholds(options);
  const wallLength = vectorLength(map.coronary_band, map.toe_tip);
  const p3Length = vectorLength(map.extensor_process, map.p3_tip);
  const scaleConfidence = Math.min(1, Math.min(wallLength, p3Length) / 18);
  const optionalCount = ['p3_heel', 'toe_ground', 'heel_ground'].filter(name => map[name]).length;
  const confidence = Math.max(0.2, Math.min(0.98, 0.55 + optionalCount * 0.1 + scaleConfidence * 0.13));

  return {
    rotation: Number(rotation.toFixed(2)),
    rawAngle: Number(p3Angle.toFixed(2)),
    referenceAngle: Number(wallAngle.toFixed(2)),
    critical: rotation > thresholds.active,
    confidence: Number(confidence.toFixed(2)),
    threshold: thresholds.active,
    thresholdMode: thresholds.view,
    status: 'landmark_measurement',
    landmarks: {
      wallStart: map.coronary_band,
      wallEnd: map.toe_tip,
      boneStart: map.extensor_process,
      boneEnd: map.p3_tip
    }
  };
}

async function imageToCanvas(imageSource, maxDimension = 1400) {
  const image = imageSource instanceof HTMLImageElement ? imageSource : await loadImage(imageSource);
  const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth || image.width, image.naturalHeight || image.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round((image.naturalWidth || image.width) * scale));
  canvas.height = Math.max(1, Math.round((image.naturalHeight || image.height) * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  return { canvas, ctx, scale };
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to load X-ray image for rotation detection.'));
    if (source instanceof Blob) image.src = URL.createObjectURL(source);
    else image.src = source;
  });
}

function detectEdgesFromCanvas(ctx, width, height) {
  const data = ctx.getImageData(0, 0, width, height).data;
  const samples = [];
  const step = Math.max(2, Math.floor(Math.max(width, height) / 240));
  for (let y = step; y < height - step; y += step) {
    for (let x = step; x < width - step; x += step) {
      const index = (y * width + x) * 4;
      const left = ((y * width + (x - step)) * 4);
      const right = ((y * width + (x + step)) * 4);
      const up = (((y - step) * width + x) * 4);
      const down = (((y + step) * width + x) * 4);
      const luminance = 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
      const gx = data[right] - data[left];
      const gy = data[down] - data[up];
      const magnitude = Math.hypot(gx, gy);
      if (magnitude > 24 && luminance > 18) samples.push({ x, y, magnitude, luminance });
    }
  }
  return samples.sort((a, b) => b.magnitude - a.magnitude);
}

function pickRegionPoint(samples, width, height, region, fallback) {
  const filtered = samples.filter(point => {
    if (region === 'dorsalTop') return point.x > width * 0.48 && point.y > height * 0.12 && point.y < height * 0.55;
    if (region === 'dorsalBottom') return point.x > width * 0.50 && point.y >= height * 0.45 && point.y < height * 0.88;
    if (region === 'boneTop') return point.x > width * 0.34 && point.x < width * 0.82 && point.y > height * 0.22 && point.y < height * 0.62;
    if (region === 'boneTip') return point.x > width * 0.45 && point.y > height * 0.42 && point.y < height * 0.86;
    return true;
  });
  const point = filtered[0] || fallback;
  return { x: point.x / width * 100, y: point.y / height * 100 };
}

async function detectImageLandmarks(imageSource) {
  const { canvas, ctx } = await imageToCanvas(imageSource);
  const samples = detectEdgesFromCanvas(ctx, canvas.width, canvas.height);
  if (samples.length < 20) throw new Error('Low contrast image: automatic landmark detection needs manual annotation.');
  return [
    { name: 'coronary_band', ...pickRegionPoint(samples, canvas.width, canvas.height, 'dorsalTop', { x: canvas.width * 0.62, y: canvas.height * 0.24 }) },
    { name: 'toe_tip', ...pickRegionPoint(samples, canvas.width, canvas.height, 'dorsalBottom', { x: canvas.width * 0.72, y: canvas.height * 0.75 }) },
    { name: 'extensor_process', ...pickRegionPoint(samples, canvas.width, canvas.height, 'boneTop', { x: canvas.width * 0.50, y: canvas.height * 0.36 }) },
    { name: 'p3_tip', ...pickRegionPoint(samples, canvas.width, canvas.height, 'boneTip', { x: canvas.width * 0.63, y: canvas.height * 0.66 }) }
  ];
}

async function detectLaminitisRotation(imageSource, options = {}) {
  if (options.landmarks?.length) return calculateRotationFromLandmarks(options.landmarks, options);
  const landmarks = await detectImageLandmarks(imageSource);
  return calculateRotationFromLandmarks(landmarks, options);
}

if (typeof window !== 'undefined') {
  window.ByrockRotation = {
    DEFAULT_THRESHOLDS,
    calculateRotationFromLandmarks,
    detectImageLandmarks,
    detectLaminitisRotation,
    getThresholds,
    normalizeAngle,
    smallestAngleBetween
  };
}

export {
  DEFAULT_THRESHOLDS,
  calculateRotationFromLandmarks,
  detectImageLandmarks,
  detectLaminitisRotation,
  getThresholds,
  normalizeAngle,
  smallestAngleBetween
};

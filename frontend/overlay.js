function ensureOverlaySvg(frame) {
  let svg = frame.querySelector('#rotationOverlaySvg');
  if (!svg) {
    svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.id = 'rotationOverlaySvg';
    svg.classList.add('rotation-overlay-svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    frame.appendChild(svg);
  }
  return svg;
}

function lineMarkup(x1, y1, x2, y2, color, width = 0.8, dash = '') {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" stroke-linecap="round" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
}

function renderRotationOverlay(frameOrCanvas, rotationData) {
  const frame = frameOrCanvas?.tagName?.toLowerCase() === 'canvas' ? frameOrCanvas.parentElement : frameOrCanvas;
  if (!frame) return;
  const svg = ensureOverlaySvg(frame);
  if (!rotationData || rotationData.status === 'needs_landmarks') {
    svg.innerHTML = '';
    return;
  }
  const color = rotationData.critical ? '#ff4757' : '#2ed573';
  const wall = rotationData.landmarks;
  const arrow = rotationData.rawAngle * Math.PI / 180;
  const center = wall?.boneStart || { x: 50, y: 38 };
  const length = 25;
  const end = wall?.boneEnd || { x: center.x + Math.cos(arrow) * length, y: center.y + Math.sin(arrow) * length };
  const labelX = Math.max(8, Math.min(86, end.x + 3));
  const labelY = Math.max(10, Math.min(90, end.y - 3));

  svg.innerHTML = `
    <defs>
      <filter id="rotationGlow"><feGaussianBlur stdDeviation="0.7" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      <marker id="rotationArrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="${color}"/></marker>
    </defs>
    ${wall ? lineMarkup(wall.wallStart.x, wall.wallStart.y, wall.wallEnd.x, wall.wallEnd.y, '#2ed573', 0.55, '2 1.4') : lineMarkup(8, 50, 92, 50, '#2ed573', 0.45, '2 1.4')}
    ${wall ? lineMarkup(wall.boneStart.x, wall.boneStart.y, wall.boneEnd.x, wall.boneEnd.y, color, 0.9) : lineMarkup(center.x, center.y, end.x, end.y, color, 0.9)}
    <line x1="${center.x}" y1="${center.y}" x2="${end.x}" y2="${end.y}" stroke="${color}" stroke-width="1.1" stroke-linecap="round" marker-end="url(#rotationArrow)" filter="url(#rotationGlow)"/>
    <text x="${labelX}" y="${labelY}" fill="${color}" font-size="4" font-family="JetBrains Mono, monospace" font-weight="700">${rotationData.rotation.toFixed(1)}°</text>
  `;
}

function clearRotationOverlay() {
  document.getElementById('rotationOverlaySvg')?.remove();
}

if (typeof window !== 'undefined') {
  window.ByrockRotationOverlay = { renderRotationOverlay, clearRotationOverlay };
}

export { renderRotationOverlay, clearRotationOverlay };

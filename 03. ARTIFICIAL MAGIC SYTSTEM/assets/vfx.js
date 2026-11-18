// ============================================================================
// MYSTIC MOTION — VFX Engine (JavaScript / Canvas 2D)
// Port of the Python OpenCV procedural visual effects engine
// ============================================================================

export const CHARACTERS = {
  "1": { key: "1", name: "DOCTOR STRANGE", color: "#009bff", accent: "#50dcff", rgb: [0, 155, 255] },
  "2": { key: "2", name: "SCARLET WITCH", color: "#eb0000", accent: "#ff2346", rgb: [235, 0, 0] },
  "3": { key: "3", name: "AGATHA HARKNESS", color: "#eb23dc", accent: "#ff91f5", rgb: [235, 35, 220] },
  "4": { key: "4", name: "LOKI", color: "#37cd2d", accent: "#9bff91", rgb: [45, 205, 55] },
  "5": { key: "5", name: "THOR", color: "#41cdff", accent: "#e1f5ff", rgb: [65, 205, 255] },
  "6": { key: "6", name: "THANOS", color: "#cd5096", accent: "#ff9bcd", rgb: [205, 80, 150] }
};

// ----------------------------------------------------------------------------
// Persistent Particle System
// ----------------------------------------------------------------------------
class Particle {
  constructor(x, y, vx, vy, life, color, radius = 2) {
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.life = life;
    this.maxLife = life;
    this.color = color;
    this.radius = radius;
  }

  update(dt) {
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.life -= dt;
    return this.life > 0;
  }

  draw(ctx) {
    const alpha = Math.max(0, Math.min(1, this.life / this.maxLife));
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = this.color;
    ctx.shadowColor = this.color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(this.x, this.y, Math.max(1, this.radius * alpha), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

export class ParticleEmitter {
  constructor(maxParticles = 350) {
    this.particles = [];
    this.maxParticles = maxParticles;
  }

  emit(x, y, vx, vy, life, color, radius = 2) {
    if (this.particles.length < this.maxParticles) {
      this.particles.push(new Particle(x, y, vx, vy, life, color, radius));
    }
  }

  updateAndDraw(ctx, dt) {
    this.particles = this.particles.filter(p => {
      const alive = p.update(dt);
      if (alive) p.draw(ctx);
      return alive;
    });
  }

  clear() {
    this.particles = [];
  }
}

// ----------------------------------------------------------------------------
// Helper Geometry Functions
// ----------------------------------------------------------------------------
export function toPx(point, width, height) {
  return {
    x: Math.max(0, Math.min(width - 1, point.x * width)),
    y: Math.max(0, Math.min(height - 1, point.y * height))
  };
}

export function palmCenter(points, width, height) {
  // Indices: 0 (wrist), 5 (index mcp), 9 (middle mcp), 13 (ring mcp), 17 (pinky mcp)
  const indices = [0, 5, 9, 13, 17];
  let sumX = 0, sumY = 0;
  for (const idx of indices) {
    sumX += points[idx].x;
    sumY += points[idx].y;
  }
  return {
    x: (sumX / indices.length) * width,
    y: (sumY / indices.length) * height
  };
}

export function handScalePx(points, width, height) {
  const p0 = points[0];
  const p9 = points[9];
  const dx = (p9.x - p0.x) * width;
  const dy = (p9.y - p0.y) * height;
  return Math.max(35, Math.hypot(dx, dy));
}

export function handOrientation(points) {
  const p0 = points[0];
  const p9 = points[9];
  return Math.atan2(p9.y - p0.y, p9.x - p0.x) + Math.PI / 2;
}

export function isHandOpen(points) {
  const wrist = points[0];
  const checks = [[8, 6], [12, 10], [16, 14], [20, 18]];
  let extended = 0;
  for (const [tip, pip] of checks) {
    const dTip = Math.hypot(points[tip].x - wrist.x, points[tip].y - wrist.y);
    const dPip = Math.hypot(points[pip].x - wrist.x, points[pip].y - wrist.y);
    if (dTip > dPip * 1.12) extended++;
  }
  return extended >= 3;
}

export function isPinching(points, width, height) {
  const dx = (points[4].x - points[8].x) * width;
  const dy = (points[4].y - points[8].y) * height;
  const dist = Math.hypot(dx, dy);
  const scale = handScalePx(points, width, height);
  return (dist / scale) < 0.28;
}

// ----------------------------------------------------------------------------
// Procedural VFX Renderers
// ----------------------------------------------------------------------------
function drawGlowCircle(ctx, x, y, radius, color, coreColor = "#ffffff") {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = radius * 1.8;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();

  if (coreColor) {
    ctx.fillStyle = coreColor;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1, radius * 0.45), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawMandala(ctx, cx, cy, radius, color, accent, t, baseAngle = 0) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.shadowColor = accent;
  ctx.shadowBlur = 12;

  // Concentric Rings
  const scales = [1.0, 0.91, 0.78, 0.64, 0.48, 0.30];
  for (let i = 0; i < scales.length; i++) {
    const r = radius * scales[i];
    const spin = (baseAngle + t * (i % 2 === 0 ? 0.6 : -0.5) + i * 0.3);
    ctx.lineWidth = i < 3 ? 2.5 : 1.5;
    ctx.strokeStyle = i % 2 === 0 ? color : accent;
    ctx.beginPath();
    ctx.arc(cx, cy, r, spin, spin + Math.PI * 1.94);
    ctx.stroke();
  }

  // 48 Radial Division Marks
  for (let i = 0; i < 48; i++) {
    const a = (Math.PI * 2 * i) / 48 + baseAngle + t * 0.05;
    const r1 = radius * (i % 4 === 0 ? 0.75 : 0.82);
    const r2 = radius * (i % 4 === 0 ? 1.04 : 0.91);
    ctx.strokeStyle = i % 4 === 0 ? accent : color;
    ctx.lineWidth = i % 4 === 0 ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
    ctx.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
    ctx.stroke();
  }

  // Nested Rotated Squares / Octagonal geometry
  const geometries = [
    { sides: 4, scale: 0.69, rot: Math.PI / 4 },
    { sides: 4, scale: 0.52, rot: 0 },
    { sides: 8, scale: 0.39, rot: Math.PI / 8 }
  ];
  for (const geom of geometries) {
    ctx.beginPath();
    for (let i = 0; i < geom.sides; i++) {
      const a = (Math.PI * 2 * i) / geom.sides + geom.rot + baseAngle + t * 0.08;
      const r = radius * geom.scale;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = color;
    ctx.stroke();
  }

  // Bright Center Seal
  drawGlowCircle(ctx, cx, cy, Math.max(4, radius * 0.12), accent, "#ffffff");
  ctx.restore();
}

function drawLightningBolt(ctx, x1, y1, x2, y3, color, branches = 2, amplitude = 16, thickness = 2.5) {
  const dx = x2 - x1;
  const dy = y3 - y1;
  const dist = Math.hypot(dx, dy) + 1e-5;
  const perpX = -dy / dist;
  const perpY = dx / dist;

  const count = Math.max(5, Math.floor(dist / 22));
  const pts = [{ x: x1, y: y1 }];

  for (let i = 1; i < count; i++) {
    const f = i / count;
    const offset = (Math.random() - 0.5) * 2 * amplitude;
    pts.push({
      x: x1 + dx * f + perpX * offset,
      y: y1 + dy * f + perpY * offset
    });
  }
  pts.push({ x: x2, y: y3 });

  ctx.save();
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
  ctx.lineWidth = thickness;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.stroke();

  // White core
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = Math.max(1, thickness * 0.45);
  ctx.stroke();

  // Branches
  if (branches > 0) {
    for (let b = 0; b < branches; b++) {
      const idx = Math.floor(1 + Math.random() * (pts.length - 2));
      const origin = pts[idx];
      const bx = origin.x + (Math.random() - 0.5) * 60;
      const by = origin.y + (Math.random() - 0.5) * 60;
      drawLightningBolt(ctx, origin.x, origin.y, bx, by, color, 0, amplitude * 0.4, Math.max(1, thickness - 1));
    }
  }
  ctx.restore();
}

function drawEnergyWisp(ctx, x1, y1, x2, y2, color, t, seed = 0, width = 3) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.hypot(dx, dy) + 1e-5;
  const perpX = -dy / dist;
  const perpY = dx / dist;

  const count = Math.max(8, Math.floor(dist / 16));
  const pts = [{ x: x1, y: y1 }];

  for (let i = 1; i < count; i++) {
    const f = i / count;
    const env = Math.sin(Math.PI * f);
    const offset = Math.sin(t * 5.0 + f * 15.0 + seed) * 14 * env
                 + Math.sin(t * 8.0 + f * 28.0 + seed) * 5 * env;
    pts.push({
      x: x1 + dx * f + perpX * offset,
      y: y1 + dy * f + perpY * offset
    });
  }
  pts.push({ x: x2, y: y2 });

  ctx.save();
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.lineWidth = width + 2;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.stroke();

  // Highlight core
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = Math.max(1, width - 1);
  ctx.stroke();
  ctx.restore();
}

// ----------------------------------------------------------------------------
// Main Magic Renderer Class
// ----------------------------------------------------------------------------
export class MagicRenderer {
  constructor() {
    this.particles = new ParticleEmitter(450);
  }

  renderStrange(ctx, palm, elapsed, power, scale, angle) {
    const radius = Math.min(180, Math.max(55, scale * 0.9 * (0.85 + 0.25 * power)));
    drawMandala(ctx, palm.x, palm.y, radius, "#ff9500", "#50dcff", elapsed, angle);

    // Drifting sparks
    for (let i = 0; i < 3; i++) {
      const a = elapsed * 1.5 + Math.random() * Math.PI * 2;
      const sx = palm.x + Math.cos(a) * radius * 1.05;
      const sy = palm.y + Math.sin(a) * radius * 1.05;
      const vx = -Math.sin(a) * (20 + Math.random() * 30);
      const vy = Math.cos(a) * (20 + Math.random() * 30);
      this.particles.emit(sx, sy, vx, vy, 0.4 + Math.random() * 0.3, "#50dcff", 2);
    }
    drawGlowCircle(ctx, palm.x, palm.y, 8, "#ff9500", "#fff5e1");
  }

  renderWanda(ctx, palm, tips, elapsed, openHand) {
    const pulse = 1.0 + 0.22 * Math.sin(elapsed * 8.0);
    drawGlowCircle(ctx, palm.x, palm.y, 24 * pulse, "#eb0000", "#ffcad4");

    // Ellipses
    ctx.save();
    ctx.strokeStyle = "#ff1a3d";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#ff1a3d";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.ellipse(palm.x, palm.y, 50, 32, elapsed * 0.8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(palm.x, palm.y, 60, 42, -elapsed * 0.65, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // Wisps to fingertips
    const activeTips = openHand ? tips.slice(1) : [tips[1]];
    activeTips.forEach((tip, idx) => {
      drawEnergyWisp(ctx, palm.x, palm.y, tip.x, tip.y, "#ff0033", elapsed + idx * 0.15, 40 + idx, 3.5);
    });

    // Orbiting embers
    for (let i = 0; i < 6; i++) {
      const a = elapsed * 0.8 + (Math.PI * 2 * i) / 6;
      const ex = palm.x + Math.cos(a) * 65;
      const ey = palm.y + Math.sin(a) * 48;
      drawGlowCircle(ctx, ex, ey, 2.5, "#ff3355", "#ffffff");
    }
  }

  renderAgatha(ctx, palm, tips, elapsed, openHand) {
    const pulse = 1.0 + 0.16 * Math.sin(elapsed * 5.5);
    drawGlowCircle(ctx, palm.x, palm.y, 22 * pulse, "#9b14b8", "#f3d5ff");

    ctx.save();
    ctx.strokeStyle = "#eb23dc";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#eb23dc";
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.ellipse(palm.x, palm.y, 42, 28, elapsed * 0.6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(palm.x, palm.y, 55, 38, -elapsed * 0.45, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    const activeTips = openHand ? tips.slice(1) : [tips[1]];
    activeTips.forEach((tip, idx) => {
      drawEnergyWisp(ctx, palm.x, palm.y, tip.x, tip.y, "#eb23dc", elapsed + idx * 0.18, 220 + idx, 3);
    });

    for (let i = 0; i < 8; i++) {
      const a = elapsed * 0.55 + (Math.PI * 2 * i) / 8;
      const rx = 60 + 8 * Math.sin(elapsed * 2 + i);
      const ry = 45 + 6 * Math.cos(elapsed * 2.3 + i);
      drawGlowCircle(ctx, palm.x + Math.cos(a) * rx, palm.y + Math.sin(a) * ry, 2.5, "#f591ff");
    }
  }

  renderLoki(ctx, palm, tips, elapsed, openHand) {
    drawGlowCircle(ctx, palm.x, palm.y, 18, "#2db33a", "#e2ffe6");

    ctx.save();
    ctx.strokeStyle = "#45df5a";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#45df5a";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.ellipse(palm.x, palm.y, 45, 30, elapsed * 0.7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    const activeTips = openHand ? tips.slice(1) : [tips[1]];
    activeTips.forEach((tip, idx) => {
      drawEnergyWisp(ctx, palm.x, palm.y, tip.x, tip.y, "#45df5a", elapsed + idx * 0.16, 320 + idx, 3);
    });

    // Rotating diamond glyphs
    for (let i = 0; i < 4; i++) {
      const a = elapsed * 0.7 + (Math.PI * 2 * i) / 4;
      const cx = palm.x + Math.cos(a) * 54;
      const cy = palm.y + Math.sin(a) * 40;
      ctx.save();
      ctx.strokeStyle = "#9bff91";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#9bff91";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(cx, cy - 7);
      ctx.lineTo(cx + 7, cy);
      ctx.lineTo(cx, cy + 7);
      ctx.lineTo(cx - 7, cy);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }
  }

  renderThor(ctx, palm, tips, elapsed, openHand) {
    const pulse = 1.0 + 0.18 * Math.sin(elapsed * 10.0);
    drawGlowCircle(ctx, palm.x, palm.y, 20 * pulse, "#3bb7ff", "#ffffff");

    const activeTips = openHand ? tips : [tips[1]];
    activeTips.forEach((tip, idx) => {
      drawLightningBolt(ctx, palm.x, palm.y, tip.x, tip.y, "#50dcff", 2, 20, 3.5);
    });

    // Upward thunder strikes
    const targets = [
      { x: palm.x - 20, y: Math.max(10, palm.y - 160) },
      { x: palm.x + 8, y: Math.max(10, palm.y - 190) },
      { x: palm.x + 40, y: Math.max(10, palm.y - 145) }
    ];
    targets.forEach(tgt => {
      drawLightningBolt(ctx, palm.x, palm.y, tgt.x, tgt.y, "#a6e8ff", 3, 26, 4);
    });
  }

  renderThanos(ctx, palm, tips, elapsed, openHand, wrist) {
    const pulse = 1.0 + 0.12 * Math.sin(elapsed * 4.5);
    drawGlowCircle(ctx, palm.x, palm.y, 22 * pulse, "#9b25cd", "#ffd6ff");

    // 6 Orbiting Infinity Stones
    const stoneColors = [
      "#ff2346", // Reality (Red)
      "#389dff", // Space (Blue)
      "#be25cd", // Power (Purple)
      "#35df5a", // Time (Green)
      "#ffea35", // Mind (Yellow)
      "#ff8825"  // Soul (Orange)
    ];
    stoneColors.forEach((color, idx) => {
      const a = elapsed * 0.5 + (Math.PI * 2 * idx) / 6;
      const sx = palm.x + Math.cos(a) * 44;
      const sy = palm.y + Math.sin(a) * 35;
      drawGlowCircle(ctx, sx, sy, 5, color, "#ffffff");
    });

    const activeTips = openHand ? tips.slice(1) : [tips[1]];
    activeTips.forEach((tip, idx) => {
      drawEnergyWisp(ctx, palm.x, palm.y, tip.x, tip.y, "#b835e0", elapsed + idx * 0.12, 820 + idx, 3);
    });

    // Gold Gauntlet Wrist Cuff
    if (wrist) {
      ctx.save();
      ctx.strokeStyle = "#e6aa28";
      ctx.lineWidth = 4;
      ctx.shadowColor = "#ffcd42";
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(wrist.x, wrist.y, 16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = "#ffe57d";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(wrist.x, wrist.y, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  renderTwoHandField(ctx, character, hands, elapsed, width, height) {
    if (hands.length < 2) return;
    const c1 = palmCenter(hands[0].points, width, height);
    const c2 = palmCenter(hands[1].points, width, height);
    const midX = (c1.x + c2.x) / 2;
    const midY = (c1.y + c2.y) / 2;
    const dist = Math.hypot(c2.x - c1.x, c2.y - c1.y);

    const name = character.name;
    if (name === "AGATHA HARKNESS") {
      ctx.save();
      ctx.strokeStyle = "#eb23dc";
      ctx.lineWidth = 18;
      ctx.shadowColor = "#f591ff";
      ctx.shadowBlur = 24;
      ctx.beginPath();
      ctx.moveTo(c1.x, c1.y);
      ctx.lineTo(c2.x, c2.y);
      ctx.stroke();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 4;
      ctx.stroke();
      ctx.restore();

      drawEnergyWisp(ctx, c1.x, c1.y, c2.x, c2.y, "#f591ff", elapsed, 701, 4);
      drawEnergyWisp(ctx, c2.x, c2.y, c1.x, c1.y, "#eb23dc", elapsed + 0.5, 702, 3);
    } else if (name === "SCARLET WITCH") {
      drawEnergyWisp(ctx, c1.x, c1.y, c2.x, c2.y, "#ff0033", elapsed, 711, 6);
      drawEnergyWisp(ctx, c2.x, c2.y, c1.x, c1.y, "#eb0028", elapsed + 0.4, 712, 4);
      drawGlowCircle(ctx, midX, midY, 16, "#ff0033", "#ffffff");
    } else if (name === "THOR") {
      for (let i = 0; i < 4; i++) {
        drawLightningBolt(ctx, c1.x, c1.y, c2.x, c2.y, "#50dcff", 2, 22 + i * 3, 3);
      }
    } else if (name === "LOKI") {
      for (let i = 0; i < 3; i++) {
        drawLightningBolt(ctx, c1.x, c1.y, c2.x, c2.y, "#45df5a", 2, 18 + i * 3, 2.5);
      }
      drawGlowCircle(ctx, midX, midY, 10, "#45df5a", "#ffffff");
    } else if (name === "DOCTOR STRANGE") {
      const radius = Math.min(Math.min(width, height) * 0.35, Math.max(60, dist * 0.46));
      drawMandala(ctx, midX, midY, radius, "#ff9500", "#50dcff", elapsed, 0);
    } else if (name === "THANOS") {
      drawEnergyWisp(ctx, c1.x, c1.y, c2.x, c2.y, "#be25cd", elapsed, 850, 4);
      drawEnergyWisp(ctx, c2.x, c2.y, c1.x, c1.y, "#50dcff", elapsed + 0.3, 851, 3);
      drawGlowCircle(ctx, midX, midY, 16, "#be25cd", "#ffd6ff");
    }
  }

  renderHand(ctx, character, hand, elapsed, width, height) {
    const palm = palmCenter(hand.points, width, height);
    const tips = [4, 8, 12, 16, 20].map(idx => toPx(hand.points[idx], width, height));
    const wrist = toPx(hand.points[0], width, height);
    const openHand = hand.gesture === "Open_Palm" || isHandOpen(hand.points);
    const pinch = isPinching(hand.points, width, height);
    const scale = handScalePx(hand.points, width, height);
    const angle = handOrientation(hand.points);
    const power = openHand ? 1.0 : (hand.gesture === "Closed_Fist" ? 0.82 : 0.68);

    const name = character.name;
    if (name === "DOCTOR STRANGE") {
      this.renderStrange(ctx, palm, elapsed, power, scale, angle);
    } else if (name === "SCARLET WITCH") {
      this.renderWanda(ctx, palm, tips, elapsed, openHand);
    } else if (name === "AGATHA HARKNESS") {
      this.renderAgatha(ctx, palm, tips, elapsed, openHand);
    } else if (name === "LOKI") {
      this.renderLoki(ctx, palm, tips, elapsed, openHand);
    } else if (name === "THOR") {
      this.renderThor(ctx, palm, tips, elapsed, openHand);
    } else if (name === "THANOS") {
      this.renderThanos(ctx, palm, tips, elapsed, openHand, wrist);
    }

    if (pinch) {
      const focusX = (tips[0].x + tips[1].x) / 2;
      const focusY = (tips[0].y + tips[1].y) / 2;
      drawGlowCircle(ctx, focusX, focusY, 6, character.accent, "#ffffff");
    }
  }

  renderAll(ctx, character, hands, elapsed, dt, width, height) {
    // Enable glowing additive screen blend for magical aura
    ctx.save();
    ctx.globalCompositeOperation = "screen";

    for (const hand of hands) {
      this.renderHand(ctx, character, hand, elapsed, width, height);
    }

    if (hands.length >= 2) {
      this.renderTwoHandField(ctx, character, hands, elapsed, width, height);
    }

    this.particles.updateAndDraw(ctx, dt);
    ctx.restore();
  }
}

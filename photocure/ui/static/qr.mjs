// Self-contained QR Code encoder (dependency-free port).
// Supports byte mode, ECC level M, versions 1-10, and emits a 2-D matrix.
// Used by the Photo Editor Studio's QR overlay (Wi-Fi, vCard, URL, Text, Phone, Email, SMS, Geo).
(function (root) {
  "use strict";

  const ECC_M = [
    // block count, total data codewords, total EC codewords (ECC level M, v1-10)
    { v: 1, blocks: 1, data: 16, ec: 10 },
    { v: 2, blocks: 1, data: 28, ec: 16 },
    { v: 3, blocks: 1, data: 44, ec: 26 },
    { v: 4, blocks: 2, data: 64, ec: 36 },
    { v: 5, blocks: 2, data: 86, ec: 48 },
    { v: 6, blocks: 4, data: 108, ec: 64 },
    { v: 7, blocks: 4, data: 124, ec: 72 },
    { v: 8, blocks: 4, data: 154, ec: 88 },
    { v: 9, blocks: 5, data: 182, ec: 110 },
    { v: 10, blocks: 5, data: 216, ec: 130 },
  ];

  const ALIGN_POS = [
    null,        // v0 unused
    [],          // v1: no alignment patterns
    [6, 18],     // v2
    [6, 22],     // v3
    [6, 26],     // v4
    [6, 30],     // v5
    [6, 34],     // v6
    [6, 22, 38], // v7
    [6, 24, 42], // v8
    [6, 26, 46], // v9
    [6, 28, 50], // v10
  ];

  // 5-bit format data = (ECC bits << 3) | mask. For level M the ECC bits are 0b00, so data == mask.
  function bchFormatBits(data5) {
    let d = data5 << 10;
    const g = 0b10100110111;
    for (let i = 14; i >= 10; i--) if (d & (1 << i)) d ^= g << (i - 10);
    return ((data5 << 10) | d) ^ 0b101010000010010;
  }

  // --- Galois field GF(256) with the QR generator polynomial ---
  const GF_EXP = new Array(512);
  const GF_LOG = new Array(256);
  (function () {
    let x = 1;
    for (let i = 0; i < 255; i++) {
      GF_EXP[i] = x;
      GF_LOG[x] = i;
      x <<= 1;
      if (x & 0x100) x ^= 0x11d;
    }
    for (let i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
  })();
  const gfMul = (a, b) => (a === 0 || b === 0 ? 0 : GF_EXP[GF_LOG[a] + GF_LOG[b]]);

  function rsGeneratorPoly(n) {
    let g = [1];
    for (let i = 0; i < n; i++) {
      const ng = new Array(g.length + 1).fill(0);
      for (let j = 0; j < g.length; j++) {
        ng[j] ^= g[j];
        ng[j + 1] ^= gfMul(g[j], GF_EXP[i]);
      }
      g = ng;
    }
    return g;
  }

  function rsEncode(data, n) {
    const gen = rsGeneratorPoly(n);
    const rem = new Array(n).fill(0);
    for (const d of data) {
      const factor = d ^ rem.shift();
      rem.push(0);
      for (let i = 0; i < n; i++) rem[i] ^= gfMul(gen[i + 1], factor);
    }
    return rem;
  }

  function versionBits(v) {
    let d = v << 12;
    const g = 0b1111100100101;
    for (let i = 17; i >= 12; i--) if (d & (1 << i)) d ^= g << (i - 12);
    return (v << 12) | d;
  }

  function pickVersion(byteLen) {
    // Return the smallest version whose byte-mode capacity fits the payload.
    for (const info of ECC_M) {
      const dataCW = info.data;
      const headerBits = 4 + (info.v <= 9 ? 8 : 16); // mode + length
      const avail = dataCW * 8 - headerBits;
      if (avail >= byteLen * 8) return info;
    }
    return null;
  }

  function buildMatrix(info) {
    const v = info.v;
    const size = 17 + 4 * v;
    const m = Array.from({ length: size }, () => new Array(size).fill(null));
    const fn = new Array(size * size).fill(false); // functional modules (no mask)

    const set = (x, y, on) => { m[y][x] = on ? 1 : 0; fn[y * size + x] = true; };

    // Finder patterns + separators
    const finder = (cx, cy) => {
      for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
        const x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= size || y >= size) continue;
        const ring = Math.max(Math.abs(dx), Math.abs(dy));
        if (ring <= 3) set(x, y, ring <= 1 || ring === 3);  // 7x7 finder: core + border, white ring 2
        else set(x, y, false);  // ring 4: white separator
      }
    };
    finder(3, 3); finder(size - 4, 3); finder(3, size - 4);

    // Timing
    for (let i = 8; i < size - 8; i++) {
      if (m[6][i] == null) { set(i, 6, i % 2 === 0); }
      if (m[i][6] == null) { set(6, i, i % 2 === 0); }
    }
    // Alignment
    const pos = ALIGN_POS[v];
    for (const ry of pos) for (const cx of pos) {
      if ((cx === 6 && ry === 6) || (cx === 6 && ry === size - 7) || (cx === size - 7 && ry === 6)) continue;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const ring = Math.max(Math.abs(dx), Math.abs(dy));
        set(cx + dx, ry + dy, ring !== 1);
      }
    }
    // Dark module is written by applyMaskAndFormat (needs the real format bits).
    for (let i = 0; i < 9; i++) { if (m[8][i] == null) set(i, 8, 0); if (m[i][8] == null) set(8, i, 0); }
    for (let i = 0; i < 8; i++) { if (m[8][size - 1 - i] == null) set(size - 1 - i, 8, 0); if (m[size - 1 - i][8] == null) set(8, size - 1 - i, 0); }
    // Version info (v >= 7): 18-bit BCH, placed per ISO 18004 (two 3x6 blocks).
    if (v >= 7) {
      const vb = versionBits(v);
      for (let i = 0; i < 18; i++) {
        const bit = (vb >> i) & 1;
        const a = Math.floor(i / 3), b = i % 3;
        set(size - 8 - 3 + b, a, bit);
        set(a, size - 8 - 3 + b, bit);
      }
    }
    return { m, fn, size };
  }

  function placeData(m, fn, size, codewords) {
    let bi = 0;
    const total = codewords.length * 8;
    let dir = -1;
    for (let x = size - 1; x >= 0; x -= 2) {
      if (x === 6) x = 5;
      for (let i = 0; i < size; i++) {
        for (let k = 0; k < 2; k++) {
          const px = x - k;
          const py = dir < 0 ? size - 1 - i : i;
          if (fn[py * size + px]) continue;
          const val = bi < total ? (codewords[bi >> 3] >> (7 - (bi & 7))) & 1 : 0;
          bi++;
          m[py][px] = val;
        }
      }
      dir = -dir;
    }
  }

  const MASKS = [
    (x, y) => (x + y) % 2 === 0,
    (x, y) => y % 2 === 0,
    (x, y) => x % 3 === 0,
    (x, y) => (x + y) % 3 === 0,
    (x, y) => (Math.floor(x / 3) + Math.floor(y / 2)) % 2 === 0,
    (x, y) => ((x * y) % 2) + ((x * y) % 3) === 0,
    (x, y) => (((x * y) % 2) + ((x * y) % 3)) % 2 === 0,
    (x, y) => (((x + y) % 2) + ((x * y) % 3)) % 2 === 0,
  ];

  function penalty(m, size) {
    let p = 0;
    for (let y = 0; y < size; y++) {
      let run = 1;
      for (let x = 1; x < size; x++) {
        if (m[y][x] === m[y][x - 1]) { run++; if (run === 5) p += 3; else if (run > 5) p++; }
        else run = 1;
      }
    }
    for (let x = 0; x < size; x++) {
      let run = 1;
      for (let y = 1; y < size; y++) {
        if (m[y][x] === m[y - 1][x]) { run++; if (run === 5) p += 3; else if (run > 5) p++; }
        else run = 1;
      }
    }
    const dark = 0x40;
    for (let y = 0; y + 1 < size; y++) for (let x = 0; x + 1 < size; x++) {
      if (m[y][x] && m[y][x + 1] && m[y + 1][x] && m[y + 1][x + 1]) p += 40;
    }
    let darkCount = 0;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if (m[y][x]) darkCount++;
    const total = size * size;
    const k = Math.max(Math.floor((darkCount - 5 * total / 100) / (total / 100)), Math.ceil((darkCount - 7 * total / 100) / (total / 100)));
    p += Math.abs(k) * 10;
    return p;
  }

  function applyMaskAndFormat(m, fn, size, mask) {
    const masked = m.map((row) => row.slice());
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      if (!fn[y * size + x] && MASKS[mask](x, y)) masked[y][x] ^= 1;
    }
    const fb = bchFormatBits(mask); // ECC level M: 5-bit data value == mask
    const bit = (i) => (fb >> i) & 1;
    // Ported verbatim from the reference `qrcode` package setupFormatInfo:
    //   vertical (column x=8): i0..i5 -> (i,8); i6,i7 -> (i+1,8); i8..i14 -> (size-15+i,8)
    //   horizontal (row y=8):  i0..i7 -> (8, size-1-i); i8 -> (8,7); i9..i14 -> (8, 14-i)
    for (let i = 0; i < 15; i++) {
      if (i < 6) masked[i][8] = bit(i);
      else if (i < 8) masked[i + 1][8] = bit(i);
      else masked[size - 15 + i][8] = bit(i);

      if (i < 8) masked[8][size - i - 1] = bit(i);
      else if (i < 9) masked[8][7] = bit(i);
      else masked[8][15 - i - 1] = bit(i);
    }
    // Fixed dark module.
    masked[size - 8][8] = 1;
    return masked;
  }

  function encode(text) {
    const bytes = new TextEncoder().encode(text);
    const info = pickVersion(bytes.length);
    if (!info) throw new Error("QR payload too long");
    const v = info.v;
    const dataCW = info.data;
    const blockCount = info.blocks;
    const totalCW = dataCW + info.ec;
    const { m, fn, size } = buildMatrix(info);

    // Bit stream: mode (4) + length (8 for v1-9 byte mode) + data
    const bits = [];
    const push = (val, n) => { for (let i = n - 1; i >= 0; i--) bits.push((val >> i) & 1); };
    push(0b0100, 4);
    push(bytes.length, v <= 9 ? 8 : 16);
    for (const b of bytes) push(b, 8);
    // Terminator
    const cap = dataCW * 8;
    const term = Math.min(4, cap - bits.length);
    push(0, term);
    while (bits.length % 8) bits.push(0);
    // Pad
    const padBytes = [0xec, 0x11];
    let pi = 0;
    while (bits.length < cap) push(padBytes[pi++ % 2], 8);

    const cw = [];
    for (let i = 0; i < bits.length; i += 8) {
      let c = 0;
      for (let k = 0; k < 8; k++) c = (c << 1) | (bits[i + k] || 0);
      cw.push(c);
    }

    // Split into blocks with the reference two-group rule (some blocks carry dataPer+1).
    const dataPer = Math.floor(dataCW / blockCount);
    const ecCount = Math.floor(totalCW / blockCount) - dataPer;
    const blocksInGroup2 = totalCW % blockCount;
    const blocksInGroup1 = blockCount - blocksInGroup2;
    const dcData = [], ecData = [];
    let off = 0;
    for (let b = 0; b < blockCount; b++) {
      const ds = b < blocksInGroup1 ? dataPer : dataPer + 1;
      const d = cw.slice(off, off + ds); off += ds;
      dcData.push(d);
      ecData.push(rsEncode(d, ecCount));
    }
    // Interleave data codewords, then EC codewords, across all blocks.
    const all = [];
    const maxData = Math.max(...dcData.map((b) => b.length));
    for (let i = 0; i < maxData; i++) for (let r = 0; r < blockCount; r++) if (i < dcData[r].length) all.push(dcData[r][i]);
    for (let i = 0; i < ecCount; i++) for (let r = 0; r < blockCount; r++) all.push(ecData[r][i]);

    placeData(m, fn, size, all.map((c) => c));

    let best = null, bestP = Infinity;
    for (let mask = 0; mask < 8; mask++) {
      const cand = applyMaskAndFormat(m, fn, size, mask);
      const p = penalty(cand, size);
      if (p < bestP) { bestP = p; best = cand; }
    }
    return best;
  }

  function toCanvas(matrix, cell = 8, margin = 4) {
    const size = matrix.length + margin * 2;
    const c = document.createElement("canvas");
    c.width = size * cell; c.height = size * cell;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#000000";
    for (let y = 0; y < matrix.length; y++) for (let x = 0; x < matrix.length; x++) {
      if (matrix[y][x]) ctx.fillRect((x + margin) * cell, (y + margin) * cell, cell, cell);
    }
    return c;
  }

  const formatters = {
    wifi: ({ ssid, password, encryption = "WPA", hidden = false }) =>
      `WIFI:S:${ssid};T:${encryption};P:${password};H:${String(!!hidden)};;`,
    vcard: (o) => `BEGIN:VCARD\nVERSION:3.0\nN:${o.name || ""}\nFN:${o.name || ""}\nORG:${o.org || ""}\nTITLE:${o.title || ""}\nTEL:${o.phone || ""}\nEMAIL:${o.email || ""}\nURL:${o.url || ""}\nEND:VCARD`,
    url: (o) => o.url,
    text: (o) => o.text,
    phone: (o) => `tel:${o.phone}`,
    email: (o) => `mailto:${o.email}${o.subject ? `?subject=${encodeURIComponent(o.subject)}` : ""}`,
    sms: (o) => `SMSTO:${o.phone}:${o.message || ""}`,
    geo: (o) => `geo:${o.lat},${o.lon}`,
  };

  root.PCUR_QR = { encode, toCanvas, formatters, version: "1.0.0" };
})(typeof window !== "undefined" ? window : globalThis);

import * as FileSystem from 'expo-file-system/legacy';

interface PlyHeader {
  format: 'ascii' | 'binary_little_endian' | 'binary_big_endian';
  vertexCount: number;
  faceCount: number;
  xIdx: number;
  yIdx: number;
  zIdx: number;
  nxIdx: number;
  nyIdx: number;
  nzIdx: number;
  rIdx: number;
  gIdx: number;
  bIdx: number;
  aIdx: number;
  vertexPropTypes: string[];
  faceCountType: string;
  faceIndexType: string;
}

function parsePlyHeader(headerText: string): PlyHeader {
  const lines = headerText.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
  let format: PlyHeader['format'] = 'ascii';
  let vertexCount = 0;
  let faceCount = 0;
  let inVertex = false;
  let inFace = false;
  const vertexProps: Array<{ name: string; type: string }> = [];
  let faceCountType = 'uchar';
  let faceIndexType = 'int';

  for (const line of lines) {
    if (line.startsWith('format ')) {
      format = line.split(/\s+/)[1] as PlyHeader['format'];
    } else if (line.startsWith('element vertex ')) {
      vertexCount = parseInt(line.split(/\s+/)[2] ?? '0', 10);
      inVertex = true;
      inFace = false;
    } else if (line.startsWith('element face ')) {
      faceCount = parseInt(line.split(/\s+/)[2] ?? '0', 10);
      inVertex = false;
      inFace = true;
    } else if (line.startsWith('element ')) {
      inVertex = false;
      inFace = false;
    } else if (line.startsWith('property ') && inVertex) {
      const parts = line.split(/\s+/);
      if (parts[1] !== 'list') {
        vertexProps.push({ name: parts[2] ?? '', type: parts[1] ?? 'float' });
      }
    } else if (line.startsWith('property list') && inFace) {
      const parts = line.split(/\s+/);
      faceCountType = parts[2] ?? 'uchar';
      faceIndexType = parts[3] ?? 'int';
    }
  }

  const findIdx = (name: string) => vertexProps.findIndex((p) => p.name === name);
  const findColor = (a: string, b: string) => {
    const i = findIdx(a);
    return i >= 0 ? i : findIdx(b);
  };

  return {
    format,
    vertexCount,
    faceCount,
    xIdx: findIdx('x'),
    yIdx: findIdx('y'),
    zIdx: findIdx('z'),
    nxIdx: findIdx('nx'),
    nyIdx: findIdx('ny'),
    nzIdx: findIdx('nz'),
    rIdx: findColor('red', 'r'),
    gIdx: findColor('green', 'g'),
    bIdx: findColor('blue', 'b'),
    aIdx: findColor('alpha', 'a'),
    vertexPropTypes: vertexProps.map((p) => p.type),
    faceCountType,
    faceIndexType,
  };
}

function typeSz(type: string): number {
  switch (type) {
    case 'double': case 'float64': return 8;
    case 'float': case 'float32': return 4;
    case 'int': case 'int32': case 'uint': case 'uint32': return 4;
    case 'short': case 'int16': case 'ushort': case 'uint16': return 2;
    case 'char': case 'int8': case 'uchar': case 'uint8': return 1;
    default: return 4;
  }
}

function readTyped(dv: DataView, offset: number, type: string, le: boolean): number {
  switch (type) {
    case 'double': case 'float64': return dv.getFloat64(offset, le);
    case 'float': case 'float32': return dv.getFloat32(offset, le);
    case 'int': case 'int32': return dv.getInt32(offset, le);
    case 'uint': case 'uint32': return dv.getUint32(offset, le);
    case 'short': case 'int16': return dv.getInt16(offset, le);
    case 'ushort': case 'uint16': return dv.getUint16(offset, le);
    case 'char': case 'int8': return dv.getInt8(offset);
    case 'uchar': case 'uint8': return dv.getUint8(offset);
    default: return dv.getFloat32(offset, le);
  }
}

function pad4(n: number): number {
  return (n + 3) & ~3;
}

// ─── Minimal PNG encoder (palette texture for vertex colors) ─────────────────
const _CRC_TABLE: Uint32Array = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[i] = c;
  }
  return t;
})();

function _crc32(data: Uint8Array): number {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) {
    const idx = ((crc ^ (data[i] as number)) & 0xff);
    crc = ((_CRC_TABLE[idx] as number) ^ (crc >>> 8)) >>> 0;
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function _pngChunk(typeStr: string, data: Uint8Array): Uint8Array {
  const tb = new TextEncoder().encode(typeStr);
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length, false);
  out.set(tb, 4);
  if (data.length > 0) out.set(data, 8);
  const src = new Uint8Array(4 + data.length);
  src.set(tb, 0);
  if (data.length > 0) src.set(data, 4);
  dv.setUint32(8 + data.length, _crc32(src), false);
  return out;
}

function _encodePngRgba(rgba: Uint8Array, width: number, height: number): Uint8Array {
  const rowBytes = width * 4;
  // Prepend filter byte 0 (None) to each scanline
  const raw = new Uint8Array(height * (1 + rowBytes));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + rowBytes)] = 0;
    raw.set(rgba.subarray(y * rowBytes, y * rowBytes + rowBytes), y * (1 + rowBytes) + 1);
  }
  // Adler32 of raw data for zlib footer
  let s1 = 1, s2 = 0;
  for (let i = 0; i < raw.length; i++) { s1 = (s1 + (raw[i] as number)) % 65521; s2 = (s2 + s1) % 65521; }
  // Deflate stored blocks (BTYPE=00, no compression)
  const MAX_BLOCK = 65534;
  const nBlocks = Math.max(1, Math.ceil(raw.length / MAX_BLOCK));
  let dLen = 2; // zlib CMF+FLG
  for (let b = 0; b < nBlocks; b++) dLen += 5 + Math.min(MAX_BLOCK, raw.length - b * MAX_BLOCK);
  dLen += 4; // Adler32
  const deflated = new Uint8Array(dLen);
  const ddv = new DataView(deflated.buffer);
  let dOff = 0;
  deflated[dOff++] = 0x78; deflated[dOff++] = 0x01; // zlib header (0x7801 % 31 == 0)
  for (let b = 0; b < nBlocks; b++) {
    const bStart = b * MAX_BLOCK;
    const bLen = Math.min(MAX_BLOCK, raw.length - bStart);
    deflated[dOff++] = b === nBlocks - 1 ? 1 : 0; // BFINAL, BTYPE=00
    ddv.setUint16(dOff, bLen, true); dOff += 2;
    ddv.setUint16(dOff, (~bLen) & 0xffff, true); dOff += 2;
    deflated.set(raw.subarray(bStart, bStart + bLen), dOff); dOff += bLen;
  }
  ddv.setUint32(dOff, ((s2 << 16) | s1) >>> 0, false); // Adler32 big-endian
  // IHDR
  const ihdrData = new Uint8Array(13);
  const idv = new DataView(ihdrData.buffer);
  idv.setUint32(0, width, false); idv.setUint32(4, height, false);
  ihdrData[8] = 8; ihdrData[9] = 6; // 8-bit RGBA
  // Assemble PNG
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = _pngChunk('IHDR', ihdrData);
  const idat = _pngChunk('IDAT', deflated);
  const iend = _pngChunk('IEND', new Uint8Array(0));
  const png = new Uint8Array(sig.length + ihdr.length + idat.length + iend.length);
  let pOff = 0;
  png.set(sig, pOff); pOff += sig.length;
  png.set(ihdr, pOff); pOff += ihdr.length;
  png.set(idat, pOff); pOff += idat.length;
  png.set(iend, pOff);
  return png;
}

function _clamp255(v: number): number { return Math.max(0, Math.min(255, Math.round(v))); }

// ─── GLB builder ─────────────────────────────────────────────────────────────
function buildGlb(
  positions: Float32Array,
  indices: Uint32Array | null,
  normals: Float32Array | null,
  colors: Float32Array | null,
): ArrayBuffer {
  const vertexCount = positions.length / 3;

  // Auto-compute smooth vertex normals from face indices if PLY has none
  let resolvedNormals = normals;
  if (!resolvedNormals && indices && indices.length > 0) {
    const computed = new Float32Array(positions.length);
    for (let i = 0; i < indices.length; i += 3) {
      const i0 = (indices[i] as number) * 3;
      const i1 = (indices[i + 1] as number) * 3;
      const i2 = (indices[i + 2] as number) * 3;
      const p0x = positions[i0] as number, p0y = positions[i0+1] as number, p0z = positions[i0+2] as number;
      const ax = (positions[i1] as number)-p0x, ay = (positions[i1+1] as number)-p0y, az = (positions[i1+2] as number)-p0z;
      const bx = (positions[i2] as number)-p0x, by = (positions[i2+1] as number)-p0y, bz = (positions[i2+2] as number)-p0z;
      const nx = ay*bz-az*by, ny = az*bx-ax*bz, nz = ax*by-ay*bx;
      computed[i0]   = (computed[i0]   as number)+nx; computed[i0+1] = (computed[i0+1] as number)+ny; computed[i0+2] = (computed[i0+2] as number)+nz;
      computed[i1]   = (computed[i1]   as number)+nx; computed[i1+1] = (computed[i1+1] as number)+ny; computed[i1+2] = (computed[i1+2] as number)+nz;
      computed[i2]   = (computed[i2]   as number)+nx; computed[i2+1] = (computed[i2+1] as number)+ny; computed[i2+2] = (computed[i2+2] as number)+nz;
    }
    for (let i = 0; i < computed.length; i += 3) {
      const cx = computed[i] as number, cy = computed[i+1] as number, cz = computed[i+2] as number;
      const len = Math.sqrt(cx*cx+cy*cy+cz*cz);
      if (len > 0) { computed[i] = cx/len; computed[i+1] = cy/len; computed[i+2] = cz/len; }
    }
    resolvedNormals = computed;
  }

  // Bounding box
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const px = positions[i] as number, py = positions[i+1] as number, pz = positions[i+2] as number;
    if (px < minX) minX = px; if (px > maxX) maxX = px;
    if (py < minY) minY = py; if (py > maxY) maxY = py;
    if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;
  }

  // Bake vertex colors → palette texture + UV coordinates
  let uvData: Float32Array | null = null;
  let pngBytes: Uint8Array | null = null;
  if (colors) {
    const texW = Math.min(256, vertexCount);
    const texH = Math.ceil(vertexCount / texW);
    const pixelData = new Uint8Array(texW * texH * 4); // default alpha=0, fill non-vertex pixels black
    uvData = new Float32Array(vertexCount * 2);
    for (let i = 0; i < vertexCount; i++) {
      const tx = i % texW;
      const ty = Math.floor(i / texW);
      const pi = (ty * texW + tx) * 4;
      pixelData[pi]   = _clamp255((colors[i*4]   as number) * 255);
      pixelData[pi+1] = _clamp255((colors[i*4+1] as number) * 255);
      pixelData[pi+2] = _clamp255((colors[i*4+2] as number) * 255);
      pixelData[pi+3] = _clamp255((colors[i*4+3] as number) * 255);
      uvData[i*2]   = (tx + 0.5) / texW;
      uvData[i*2+1] = (ty + 0.5) / texH;
    }
    pngBytes = _encodePngRgba(pixelData, texW, texH);
  }

  // Binary data slices
  const posBytes  = new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength);
  const normBytes = resolvedNormals ? new Uint8Array(resolvedNormals.buffer, resolvedNormals.byteOffset, resolvedNormals.byteLength) : null;
  const uvBytes   = uvData   ? new Uint8Array(uvData.buffer,   uvData.byteOffset,   uvData.byteLength)   : null;
  const idxBytes  = indices  ? new Uint8Array(indices.buffer,  indices.byteOffset,  indices.byteLength)  : null;

  // Buffer layout
  let binOff = 0;
  const posViewOffset  = binOff; binOff += pad4(posBytes.length);
  const normViewOffset = normBytes ? binOff : -1; if (normBytes) binOff += pad4(normBytes.length);
  const uvViewOffset   = uvBytes   ? binOff : -1; if (uvBytes)   binOff += pad4(uvBytes.length);
  const idxViewOffset  = idxBytes  ? binOff : -1; if (idxBytes)  binOff += pad4(idxBytes.length);
  const imgViewOffset  = pngBytes  ? binOff : -1; if (pngBytes)  binOff += pad4(pngBytes.length);
  const totalBinBytes  = binOff;

  // glTF structures
  const bufferViews: object[] = [];
  const accessors:   object[] = [];
  const attributes: Record<string, number> = {};

  bufferViews.push({ buffer: 0, byteOffset: posViewOffset, byteLength: posBytes.length, target: 34962 });
  accessors.push({ bufferView: 0, componentType: 5126, count: vertexCount, type: 'VEC3', min: [minX,minY,minZ], max: [maxX,maxY,maxZ] });
  attributes['POSITION'] = 0;

  if (normBytes && normViewOffset >= 0) {
    bufferViews.push({ buffer: 0, byteOffset: normViewOffset, byteLength: normBytes.length, target: 34962 });
    accessors.push({ bufferView: bufferViews.length-1, componentType: 5126, count: vertexCount, type: 'VEC3' });
    attributes['NORMAL'] = accessors.length-1;
  }

  if (uvBytes && uvViewOffset >= 0) {
    bufferViews.push({ buffer: 0, byteOffset: uvViewOffset, byteLength: uvBytes.length, target: 34962 });
    accessors.push({ bufferView: bufferViews.length-1, componentType: 5126, count: vertexCount, type: 'VEC2' });
    attributes['TEXCOORD_0'] = accessors.length-1;
  }

  let indicesAccessorIdx: number | undefined;
  if (idxBytes && indices && idxViewOffset >= 0) {
    bufferViews.push({ buffer: 0, byteOffset: idxViewOffset, byteLength: idxBytes.length, target: 34963 });
    accessors.push({ bufferView: bufferViews.length-1, componentType: 5125, count: indices.length, type: 'SCALAR' });
    indicesAccessorIdx = accessors.length-1;
  }

  let imgBvIdx = -1;
  if (pngBytes && imgViewOffset >= 0) {
    imgBvIdx = bufferViews.length;
    bufferViews.push({ buffer: 0, byteOffset: imgViewOffset, byteLength: pngBytes.length });
  }

  const hasTexture = imgBvIdx >= 0;
  const material: Record<string, unknown> = {
    pbrMetallicRoughness: {
      baseColorFactor: [1.0, 1.0, 1.0, 1.0],
      ...(hasTexture ? { baseColorTexture: { index: 0 } } : {}),
      metallicFactor: 0.0,
      roughnessFactor: 1.0,
    },
    extensions: { KHR_materials_unlit: {} },
    doubleSided: true,
  };

  const gltf: Record<string, unknown> = {
    asset: { version: '2.0', generator: 'VRoom' },
    extensionsUsed: ['KHR_materials_unlit'],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes, material: 0, mode: indicesAccessorIdx !== undefined ? 4 : 0, ...(indicesAccessorIdx !== undefined ? { indices: indicesAccessorIdx } : {}) }] }],
    materials: [material],
    accessors,
    bufferViews,
    buffers: [{ byteLength: totalBinBytes }],
  };

  if (hasTexture) {
    gltf['images']   = [{ bufferView: imgBvIdx, mimeType: 'image/png' }];
    gltf['samplers'] = [{ magFilter: 9728, minFilter: 9728, wrapS: 33071, wrapT: 33071 }]; // NEAREST, CLAMP_TO_EDGE
    gltf['textures'] = [{ sampler: 0, source: 0 }];
  }

  const jsonStr = JSON.stringify(gltf);
  const jsonPadded = jsonStr.padEnd(pad4(jsonStr.length), ' ');
  const jsonEncoded = new TextEncoder().encode(jsonPadded);

  const totalLen = 12 + 8 + jsonEncoded.length + (totalBinBytes > 0 ? 8 + totalBinBytes : 0);
  const glb = new ArrayBuffer(totalLen);
  const dv = new DataView(glb);
  const out = new Uint8Array(glb);
  let off = 0;

  dv.setUint32(off, 0x46546C67, true); off += 4; // "glTF"
  dv.setUint32(off, 2, true);          off += 4;
  dv.setUint32(off, totalLen, true);   off += 4;
  dv.setUint32(off, jsonEncoded.length, true); off += 4;
  dv.setUint32(off, 0x4E4F534A, true);         off += 4; // "JSON"
  out.set(jsonEncoded, off); off += jsonEncoded.length;

  if (totalBinBytes > 0) {
    dv.setUint32(off, totalBinBytes, true); off += 4;
    dv.setUint32(off, 0x004E4942, true);    off += 4; // "BIN\0"
    const bs = off;
    out.set(posBytes, bs + posViewOffset);
    if (normBytes && normViewOffset >= 0) out.set(normBytes, bs + normViewOffset);
    if (uvBytes   && uvViewOffset   >= 0) out.set(uvBytes,   bs + uvViewOffset);
    if (idxBytes  && idxViewOffset  >= 0) out.set(idxBytes,  bs + idxViewOffset);
    if (pngBytes  && imgViewOffset  >= 0) out.set(pngBytes,  bs + imgViewOffset);
  }

  return glb;
}

function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let str = '';
  for (let i = 0; i < bytes.length; i++) str += String.fromCharCode(bytes[i] as number);
  return btoa(str);
}

export async function convertPlyToGlb(plyUri: string, outputGlbUri: string): Promise<void> {
  // Read file as base64 — safe for both ASCII and binary PLY
  const b64 = await FileSystem.readAsStringAsync(plyUri, {
    encoding: FileSystem.EncodingType.Base64,
  });
  const binStr = atob(b64);
  const bytes = new Uint8Array(binStr.length);
  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);

  // Find "end_header" marker
  const marker = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114]; // "end_header"
  let headerEnd = -1;
  outer: for (let i = 0; i <= bytes.length - marker.length; i++) {
    for (let j = 0; j < marker.length; j++) {
      if (bytes[i + j] !== marker[j]) continue outer;
    }
    headerEnd = i + marker.length;
    if (bytes[headerEnd] === 0x0d) headerEnd++; // \r
    if (bytes[headerEnd] === 0x0a) headerEnd++; // \n
    break;
  }
  if (headerEnd === -1) throw new Error('Invalid PLY: no end_header found');

  const headerText = new TextDecoder().decode(bytes.subarray(0, headerEnd));
  const header = parsePlyHeader(headerText);
  const data = bytes.subarray(headerEnd);

  if (header.xIdx < 0 || header.yIdx < 0 || header.zIdx < 0) {
    throw new Error('PLY file has no x/y/z vertex properties');
  }

  const hasNormals = header.nxIdx >= 0 && header.nyIdx >= 0 && header.nzIdx >= 0;
  const hasColors = header.rIdx >= 0 && header.gIdx >= 0 && header.bIdx >= 0;
  const positions = new Float32Array(header.vertexCount * 3);
  const normals = hasNormals ? new Float32Array(header.vertexCount * 3) : null;
  // Colors stored as RGBA float (VEC4)
  const colors = hasColors ? new Float32Array(header.vertexCount * 4) : null;

  if (header.format === 'ascii') {
    const textLines = new TextDecoder().decode(data).split('\n');
    for (let i = 0; i < header.vertexCount; i++) {
      const parts = textLines[i]?.trim().split(/\s+/).map(Number) ?? [];
      positions[i * 3]     = parts[header.xIdx] ?? 0;
      positions[i * 3 + 1] = parts[header.yIdx] ?? 0;
      positions[i * 3 + 2] = parts[header.zIdx] ?? 0;
      if (normals) {
        normals[i * 3]     = parts[header.nxIdx] ?? 0;
        normals[i * 3 + 1] = parts[header.nyIdx] ?? 0;
        normals[i * 3 + 2] = parts[header.nzIdx] ?? 0;
      }
      if (colors) {
        const rType = header.vertexPropTypes[header.rIdx] ?? 'uchar';
        const scale = (rType === 'uchar' || rType === 'uint8') ? 1 / 255 : 1;
        colors[i * 4]     = ((parts[header.rIdx] as number | undefined) ?? 0) * scale;
        colors[i * 4 + 1] = ((parts[header.gIdx] as number | undefined) ?? 0) * scale;
        colors[i * 4 + 2] = ((parts[header.bIdx] as number | undefined) ?? 0) * scale;
        colors[i * 4 + 3] = header.aIdx >= 0 ? ((parts[header.aIdx] as number | undefined) ?? 255) * scale : 1.0;
      }
    }

    let indices: Uint32Array | null = null;
    if (header.faceCount > 0) {
      const triIdx: number[] = [];
      for (let i = 0; i < header.faceCount; i++) {
        const parts = textLines[header.vertexCount + i]?.trim().split(/\s+/).map(Number) ?? [];
        const cnt = (parts[0] as number | undefined) ?? 0;
        for (let j = 1; j < cnt - 1; j++) {
          triIdx.push(
            (parts[1] as number | undefined) ?? 0,
            (parts[j + 1] as number | undefined) ?? 0,
            (parts[j + 2] as number | undefined) ?? 0,
          );
        }
      }
      if (triIdx.length > 0) indices = new Uint32Array(triIdx);
    }

    const glb = buildGlb(positions, indices, normals, colors);
    await FileSystem.writeAsStringAsync(outputGlbUri, arrayBufferToBase64(glb), {
      encoding: FileSystem.EncodingType.Base64,
    });
    return;
  }

  // Binary PLY
  const le = header.format === 'binary_little_endian';
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);

  const propOffsets: number[] = [];
  let stride = 0;
  for (const t of header.vertexPropTypes) {
    propOffsets.push(stride);
    stride += typeSz(t);
  }

  for (let i = 0; i < header.vertexCount; i++) {
    const base = i * stride;
    const xOff = propOffsets[header.xIdx] as number ?? 0;
    const yOff = propOffsets[header.yIdx] as number ?? 0;
    const zOff = propOffsets[header.zIdx] as number ?? 0;
    const xType = header.vertexPropTypes[header.xIdx] ?? 'float';
    const yType = header.vertexPropTypes[header.yIdx] ?? 'float';
    const zType = header.vertexPropTypes[header.zIdx] ?? 'float';
    positions[i * 3]     = readTyped(dv, base + xOff, xType, le);
    positions[i * 3 + 1] = readTyped(dv, base + yOff, yType, le);
    positions[i * 3 + 2] = readTyped(dv, base + zOff, zType, le);
    if (normals) {
      const nxOff = propOffsets[header.nxIdx] as number ?? 0;
      const nyOff = propOffsets[header.nyIdx] as number ?? 0;
      const nzOff = propOffsets[header.nzIdx] as number ?? 0;
      normals[i * 3]     = readTyped(dv, base + nxOff, header.vertexPropTypes[header.nxIdx] ?? 'float', le);
      normals[i * 3 + 1] = readTyped(dv, base + nyOff, header.vertexPropTypes[header.nyIdx] ?? 'float', le);
      normals[i * 3 + 2] = readTyped(dv, base + nzOff, header.vertexPropTypes[header.nzIdx] ?? 'float', le);
    }
    if (colors) {
      const rType = header.vertexPropTypes[header.rIdx] ?? 'uchar';
      const scale = (rType === 'uchar' || rType === 'uint8') ? 1 / 255 : 1;
      colors[i * 4]     = readTyped(dv, base + (propOffsets[header.rIdx] as number ?? 0), rType, le) * scale;
      colors[i * 4 + 1] = readTyped(dv, base + (propOffsets[header.gIdx] as number ?? 0), header.vertexPropTypes[header.gIdx] ?? 'uchar', le) * scale;
      colors[i * 4 + 2] = readTyped(dv, base + (propOffsets[header.bIdx] as number ?? 0), header.vertexPropTypes[header.bIdx] ?? 'uchar', le) * scale;
      colors[i * 4 + 3] = header.aIdx >= 0
        ? readTyped(dv, base + (propOffsets[header.aIdx] as number ?? 0), header.vertexPropTypes[header.aIdx] ?? 'uchar', le) * scale
        : 1.0;
    }
  }

  let indices: Uint32Array | null = null;
  if (header.faceCount > 0) {
    let fOff = header.vertexCount * stride;
    const triIdx: number[] = [];
    const cntSz = typeSz(header.faceCountType);
    const idxSz = typeSz(header.faceIndexType);

    for (let i = 0; i < header.faceCount; i++) {
      const cnt = readTyped(dv, fOff, header.faceCountType, le); fOff += cntSz;
      const verts: number[] = [];
      for (let j = 0; j < cnt; j++) {
        verts.push(readTyped(dv, fOff, header.faceIndexType, le)); fOff += idxSz;
      }
      for (let j = 1; j < cnt - 1; j++) {
        triIdx.push(
          verts[0] as number ?? 0,
          verts[j] as number ?? 0,
          verts[j + 1] as number ?? 0,
        );
      }
    }
    if (triIdx.length > 0) indices = new Uint32Array(triIdx);
  }

  const glb = buildGlb(positions, indices, normals, colors);
  await FileSystem.writeAsStringAsync(outputGlbUri, arrayBufferToBase64(glb), {
    encoding: FileSystem.EncodingType.Base64,
  });
}

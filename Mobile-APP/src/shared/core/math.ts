// ────────────────────────────────────────────────────────────
// VRoom Math Utilities
// ────────────────────────────────────────────────────────────
import { Vec3 } from './types';

/** Dot product of two Vec3 vectors */
export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Euclidean distance between two Vec3 points */
export function distance(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Length (magnitude) of a Vec3 */
export function length(v: Vec3): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/** Normalise a Vec3. Returns [0,0,0] if the vector has zero length. */
export function normalize(v: Vec3): Vec3 {
  const len = length(v);
  if (len === 0) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

/** Subtract vector b from vector a:  a − b */
export function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** Cross product of two Vec3 vectors */
export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * Cosine similarity between two direction vectors.
 * Returns value in [-1, 1]. 1 = identical direction, -1 = opposite.
 */
export function cosineSimilarity(a: Vec3, b: Vec3): number {
  const na = normalize(a);
  const nb = normalize(b);
  return dot(na, nb);
}

import { Matrix4, Vec3 } from './types';

function normalize(v: Vec3): Vec3 {
  const mag = Math.hypot(v[0], v[1], v[2]);
  if (mag < 1e-8) {
    return [0, 0, 0];
  }
  return [v[0] / mag, v[1] / mag, v[2] / mag];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function multiply4x4(a: Matrix4, b: Matrix4): Matrix4 {
  const out: number[][] = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      out[row]![col] =
        a[row]![0]! * b[0]![col]! +
        a[row]![1]! * b[1]![col]! +
        a[row]![2]! * b[2]![col]! +
        a[row]![3]! * b[3]![col]!;
    }
  }
  return out as Matrix4;
}

export function invertRigidTransform(m: Matrix4): Matrix4 {
  const r00 = m[0][0];
  const r01 = m[0][1];
  const r02 = m[0][2];
  const r10 = m[1][0];
  const r11 = m[1][1];
  const r12 = m[1][2];
  const r20 = m[2][0];
  const r21 = m[2][1];
  const r22 = m[2][2];
  const tx = m[0][3];
  const ty = m[1][3];
  const tz = m[2][3];

  return [
    [r00, r10, r20, -(r00 * tx + r10 * ty + r20 * tz)],
    [r01, r11, r21, -(r01 * tx + r11 * ty + r21 * tz)],
    [r02, r12, r22, -(r02 * tx + r12 * ty + r22 * tz)],
    [0, 0, 0, 1],
  ];
}

export function makeCameraToWorld(position: Vec3, forward: Vec3, up: Vec3): Matrix4 {
  const forwardUnit = normalize(forward);
  const upUnit = normalize(up);
  const right = normalize(cross(forwardUnit, upUnit));
  const correctedUp = normalize(cross([-forwardUnit[0], -forwardUnit[1], -forwardUnit[2]], right));
  const backward: Vec3 = [-forwardUnit[0], -forwardUnit[1], -forwardUnit[2]];

  return [
    [right[0], correctedUp[0], backward[0], position[0]],
    [right[1], correctedUp[1], backward[1], position[1]],
    [right[2], correctedUp[2], backward[2], position[2]],
    [0, 0, 0, 1],
  ];
}

export function rebaseToSessionRoot(rootInverse: Matrix4, cameraToWorld: Matrix4): Matrix4 {
  return multiply4x4(rootInverse, cameraToWorld);
}

export function estimateIntrinsicsFromVerticalFov(
  width: number,
  height: number,
  diagonalFovDeg: number,
): { fx: number; fy: number; cx: number; cy: number } {
  const diagonal = Math.hypot(width, height);
  const diagonalFovRad = (diagonalFovDeg * Math.PI) / 180;
  const focal = diagonal / (2 * Math.tan(diagonalFovRad / 2));

  return {
    fx: focal,
    fy: focal,
    cx: width / 2,
    cy: height / 2,
  };
}

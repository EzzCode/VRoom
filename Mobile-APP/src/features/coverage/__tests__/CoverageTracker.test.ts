import { CoverageTracker } from '../CoverageTracker';
import { CameraPose } from '../../../shared/core/types';

function makePose(overrides: Partial<CameraPose> = {}): CameraPose {
  return {
    position: [0, 0, 0],
    rotation: [0, 0, 0],
    forward: [0, 0, -1],
    up: [0, 1, 0],
    timestamp: 0,
    ...overrides,
  };
}

describe('CoverageTracker', () => {
  const baseConfig = {
    voxelSize: 0.5,
    minObservations: 2,
    fovDeg: 60,
    frustumDepth: 2,
  };

  it('starts empty with 0% coverage', () => {
    const t = new CoverageTracker(baseConfig);
    expect(t.coveragePercent).toBe(0);
    expect(t.touchedVoxelCount).toBe(0);
    expect(t.coveredVoxelCount).toBe(0);
  });

  it('first observe touches voxels but covers none (min=2)', () => {
    const t = new CoverageTracker(baseConfig);
    const r = t.observe(makePose());
    expect(r.newlyTouched).toBeGreaterThan(0);
    expect(r.newlyCovered).toBe(0);
    expect(t.coveredVoxelCount).toBe(0);
    expect(t.coveragePercent).toBe(0);
  });

  it('observing the same pose twice covers the voxels', () => {
    const t = new CoverageTracker(baseConfig);
    t.observe(makePose());
    const r2 = t.observe(makePose());
    expect(r2.newlyCovered).toBeGreaterThan(0);
    expect(t.coveredVoxelCount).toBe(t.touchedVoxelCount);
    expect(t.coveragePercent).toBe(1);
  });

  it('a pose looking the other way touches different voxels', () => {
    const t = new CoverageTracker(baseConfig);
    const r1 = t.observe(makePose({ forward: [0, 0, -1] }));
    const r2 = t.observe(makePose({ forward: [0, 0, 1] }));
    const set1 = new Set(r1.observedKeys);
    const overlap = r2.observedKeys.filter((k) => set1.has(k));
    expect(overlap.length).toBeLessThan(r2.observedKeys.length);
  });

  it('peek does not mutate state', () => {
    const t = new CoverageTracker(baseConfig);
    const peeked = t.peek(makePose());
    expect(peeked.newlyTouched).toBeGreaterThan(0);
    expect(t.touchedVoxelCount).toBe(0);
    expect(t.coveredVoxelCount).toBe(0);
  });

  it('reset clears all state', () => {
    const t = new CoverageTracker(baseConfig);
    t.observe(makePose());
    t.observe(makePose());
    expect(t.touchedVoxelCount).toBeGreaterThan(0);
    t.reset();
    expect(t.touchedVoxelCount).toBe(0);
    expect(t.coveredVoxelCount).toBe(0);
    expect(t.coveragePercent).toBe(0);
  });

  it('getVoxels returns world-centred voxel views', () => {
    const t = new CoverageTracker(baseConfig);
    t.observe(makePose());
    const voxels = t.getVoxels();
    expect(voxels.length).toBe(t.touchedVoxelCount);
    expect(voxels.every((v) => v.state === 'partial')).toBe(true);
    // Camera looks down -Z, so voxel centres should mostly have negative z.
    const negZ = voxels.filter((v) => v.center[2] < 0).length;
    expect(negZ).toBeGreaterThan(voxels.length / 2);
  });
});

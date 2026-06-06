// ────────────────────────────────────────────────────────────
// VoxelOverlay — renders CoverageTracker voxels inside a ViroARScene
// ────────────────────────────────────────────────────────────
//
// Each voxel becomes a translucent ViroBox at its world centre, coloured
// by state (yellow = partial, green = covered). Must be mounted inside a
// `ViroARScene` because ViroNode children require an AR/3D scene parent.
// ────────────────────────────────────────────────────────────

import React, { useMemo, useRef } from 'react';
import { ViroBox, ViroMaterials, ViroNode } from '@reactvision/react-viro';
import { VoxelView } from './CoverageTracker';

// Register materials once for the whole app.
ViroMaterials.createMaterials({
  voxelPartial: {
    diffuseColor: '#FFD60088', // yellow w/ ~50% alpha
    lightingModel: 'Constant',
  },
  voxelCovered: {
    diffuseColor: '#34D39988', // green w/ ~50% alpha
    lightingModel: 'Constant',
  },
});

export interface VoxelOverlayProps {
  voxels: VoxelView[];
  /** Edge length of each rendered cube, in metres. Should match tracker config. */
  voxelSize: number;
  /** Optional cap to avoid rendering thousands of boxes on weak devices. */
  maxRender?: number;
}

const VoxelOverlay: React.FC<VoxelOverlayProps> = ({ voxels, voxelSize, maxRender = 800 }) => {
  // Slightly smaller than the voxel cell so adjacent voxels don't z-fight.
  const renderSize = voxelSize * 0.85;
  const warnedRef = useRef(false);

  // When over the cap, keep every covered (green) voxel first — those are the
  // meaningful result — then fill the remaining budget with the most recently
  // touched partials. A plain `slice(-maxRender)` would drop covered voxels by
  // age and make coverage visually "disappear".
  const limited = useMemo(() => {
    if (voxels.length <= maxRender) return voxels;
    if (!warnedRef.current) {
      warnedRef.current = true;
      console.warn(
        `[VoxelOverlay] ${voxels.length} voxels exceeds maxRender=${maxRender}; ` +
          'rendering covered voxels + most recent partials.',
      );
    }
    const covered = voxels.filter((v) => v.state === 'covered');
    const partial = voxels.filter((v) => v.state !== 'covered');
    const budget = Math.max(0, maxRender - covered.length);
    // covered may itself exceed the cap on dense scans — keep the most recent.
    const keptCovered = covered.length > maxRender ? covered.slice(-maxRender) : covered;
    return keptCovered.concat(partial.slice(-budget));
  }, [voxels, maxRender]);

  return (
    <ViroNode>
      {limited.map((v) => (
        <ViroBox
          key={v.key}
          position={v.center}
          width={renderSize}
          height={renderSize}
          length={renderSize}
          materials={[v.state === 'covered' ? 'voxelCovered' : 'voxelPartial']}
        />
      ))}
    </ViroNode>
  );
};

export default VoxelOverlay;

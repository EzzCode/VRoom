// ────────────────────────────────────────────────────────────
// VoxelOverlay — renders CoverageTracker voxels inside a ViroARScene
// ────────────────────────────────────────────────────────────
//
// Each voxel becomes a translucent ViroBox at its world centre, coloured
// by state (yellow = partial, green = covered). Must be mounted inside a
// `ViroARScene` because ViroNode children require an AR/3D scene parent.
// ────────────────────────────────────────────────────────────

import React, { useMemo, useEffect } from 'react';
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

const VoxelOverlay: React.FC<VoxelOverlayProps> = ({ voxels, voxelSize, maxRender = 400 }) => {
  // Slightly smaller than the voxel cell so adjacent voxels don't z-fight.
  const renderSize = voxelSize * 0.85;

  const limited = useMemo(
    () => (voxels.length > maxRender ? voxels.slice(0, maxRender) : voxels),
    [voxels, maxRender],
  );

  useEffect(() => {
    if (voxels.length > maxRender) {
      console.warn(
        `[VoxelOverlay] ${voxels.length} voxels exceeds maxRender=${maxRender}, truncating.`,
      );
    }
  }, [voxels.length, maxRender]);

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

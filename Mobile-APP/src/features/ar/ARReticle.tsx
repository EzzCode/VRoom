import React from 'react';
import { ViroNode, ViroQuad } from '@reactvision/react-viro';

interface ARReticleProps {
  position: [number, number, number];
  visible: boolean;
}

export default function ARReticle({ position, visible }: ARReticleProps) {
  if (!visible) return null;

  return (
    <ViroNode position={position} rotation={[-90, 0, 0]}>
      <ViroQuad
        position={[0, 0, 0.001]}
        width={0.05}
        height={0.05}
        materials={['reticleMaterial']}
        arShadowReceiver={false}
      />
    </ViroNode>
  );
}

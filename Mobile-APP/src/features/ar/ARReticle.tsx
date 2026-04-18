import { ViroNode, ViroImage, ViroAnimations } from '@reactvision/react-viro';

ViroAnimations.registerAnimations({
  pulse: {
    properties: {
      scaleX: 1.1,
      scaleY: 1.1,
      scaleZ: 1.1,
      opacity: 0.8,
    },
    easing: 'EaseInEaseOut',
    duration: 1000,
  },
  reset: {
    properties: {
      scaleX: 1.0,
      scaleY: 1.0,
      scaleZ: 1.0,
      opacity: 1.0,
    },
    easing: 'EaseInEaseOut',
    duration: 1000,
  },
  pulseLoop: ['pulse', 'reset'] as any,
});

interface ARReticleProps {
  position: [number, number, number];
  visible: boolean;
}

export default function ARReticle({ position, visible }: ARReticleProps) {
  if (!visible) return null;

  return (
    <ViroNode position={position} rotation={[-90, 0, 0]}>
      <ViroImage
        source={require('../../../assets/meshes/ar_reticle.png')}
        width={0.15}
        height={0.15}
        animation={{
          name: 'pulseLoop',
          run: true,
          loop: true,
        }}
      />
    </ViroNode>
  );
}

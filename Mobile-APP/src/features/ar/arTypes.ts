/**
 * Shared AR types used by ARViewScreen and ARMeshScene.
 * Positions are NOT stored here — ARMeshScene owns them for performance
 * (avoids parent re-renders on every drag event).
 */
export interface PlacedMesh {
  id: string;
  meshSource: any;
  meshType: 'GLB' | 'OBJ';
  meshName: string;
  /** Current rotation [x, y, z] in degrees — owned by ARViewScreen */
  rotation: [number, number, number];
  /** Current uniform scale [x, y, z] — owned by ARViewScreen */
  scale: [number, number, number];
  /** Whether the mesh has been placed in the AR scene yet */
  isPlaced: boolean;
}

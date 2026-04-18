import * as FileSystem from 'expo-file-system/legacy';
import * as DocumentPicker from 'expo-document-picker';
import { MeshInfo } from '../../shared/core/types';

const IMPORTED_MESH_DIR = `${FileSystem.documentDirectory}meshes/`;

const SUPER_CLEAN_GLB = require('../../../assets/meshes/super_clean.glb');

const BUNDLED_MESHES: MeshInfo[] = [
  {
    id: 'bundled_super_clean',
    name: 'Super Clean',
    format: 'GLB',
    size: 928736,
    uri: 'bundled://super_clean',
    isBundled: true,
  },
];

export function getBundledMeshes(): MeshInfo[] {
  return BUNDLED_MESHES;
}

export async function getImportedMeshes(): Promise<MeshInfo[]> {
  const dirInfo = await FileSystem.getInfoAsync(IMPORTED_MESH_DIR);
  if (!dirInfo.exists) {
    return [];
  }

  const files = await FileSystem.readDirectoryAsync(IMPORTED_MESH_DIR);
  const meshes: MeshInfo[] = [];

  for (const file of files) {
    const ext = file.split('.').pop()?.toUpperCase();
    if (ext !== 'GLB' && ext !== 'OBJ') continue;

    const filePath = `${IMPORTED_MESH_DIR}${file}`;
    const info = await FileSystem.getInfoAsync(filePath);
    const name = file.replace(/\.(glb|obj)$/i, '');

    const fileUri = filePath.startsWith('file://') ? filePath : `file://${filePath}`;

    meshes.push({
      id: `imported_${file}`,
      name,
      format: ext as 'GLB' | 'OBJ',
      size: (info as any).size ?? 0,
      uri: fileUri,
      isBundled: false,
    });
  }

  return meshes;
}

export async function getAvailableMeshes(): Promise<MeshInfo[]> {
  const bundled = getBundledMeshes();
  const imported = await getImportedMeshes();
  return [...bundled, ...imported];
}

export async function importMeshFromFilePicker(): Promise<MeshInfo | null> {
  const result = await DocumentPicker.getDocumentAsync({
    type: ['model/gltf-binary', 'model/gltf+json', 'model/obj', '*/*'],
    copyToCacheDirectory: true,
  });

  if (result.canceled || !result.assets?.length) {
    return null;
  }

  const asset = result.assets?.[0];
  if (!asset) return null;

  const fileName = asset.name;
  const ext = fileName.split('.').pop()?.toUpperCase();

  if (ext !== 'GLB' && ext !== 'OBJ') {
    return null;
  }

  if (!asset.uri) return null;

  await FileSystem.makeDirectoryAsync(IMPORTED_MESH_DIR, { intermediates: true });

  const destPath = `${IMPORTED_MESH_DIR}${fileName}`;
  await FileSystem.copyAsync({
    from: asset.uri,
    to: destPath,
  });

  const name = fileName.replace(/\.(glb|obj)$/i, '');
  const info = await FileSystem.getInfoAsync(destPath);

  return {
    id: `imported_${fileName}`,
    name,
    format: ext as 'GLB' | 'OBJ',
    size: (info as any).size ?? 0,
    uri: destPath,
    isBundled: false,
  };
}

export async function deleteImportedMesh(mesh: MeshInfo): Promise<void> {
  if (mesh.isBundled) return;
  await FileSystem.deleteAsync(mesh.uri, { idempotent: true });
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

export function getMeshSource(mesh: MeshInfo): {
  source: number | { uri: string };
  type: 'GLB' | 'OBJ';
} {
  if (mesh.isBundled) {
    return {
      source: SUPER_CLEAN_GLB,
      type: mesh.format,
    };
  }
  const uri = mesh.uri.startsWith('file://') ? mesh.uri : `file://${mesh.uri}`;
  return {
    source: { uri },
    type: mesh.format,
  };
}

import * as FileSystem from 'expo-file-system/legacy';
import * as DocumentPicker from 'expo-document-picker';
import { MeshInfo } from '../../shared/core/types';
import { convertPlyToGlb } from './plyToGlb';

const IMPORTED_MESH_DIR = `${FileSystem.documentDirectory}meshes/`;

// Metro dev server URL reachable from device via `adb reverse tcp:8083 tcp:8083`.
// Used to serve runtime-uploaded GLBs over HTTP because ViroReact's native loader
// fails on file:// URIs to the app sandbox storage.
const METRO_BASE_URL = 'http://127.0.0.1:8083';

/**
 * Uploads a local GLB file to the Metro dev server and returns the http:// URL
 * ViroReact can fetch it from. Required because file:// URIs to internal
 * storage fail in ViroReact's native GLTF loader.
 */
export async function uploadMeshToMetro(localPath: string, id: string): Promise<string> {
  const path = localPath.startsWith('file://') ? localPath : `file://${localPath}`;
  const base64 = await FileSystem.readAsStringAsync(path, {
    encoding: FileSystem.EncodingType.Base64,
  });
  const safeId = id.replace(/[^a-zA-Z0-9_-]/g, '_');
  const resp = await fetch(`${METRO_BASE_URL}/dynamic-mesh/upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: safeId, base64 }),
  });
  if (!resp.ok) {
    throw new Error(`Metro upload failed: ${resp.status} ${await resp.text()}`);
  }
  return `${METRO_BASE_URL}/dynamic-mesh/${safeId}.glb`;
}

/**
 * Ensures the given mesh has an http:// URI ViroReact can load. For bundled
 * meshes this is a no-op. For imported meshes with file:// or local paths, the
 * file is uploaded to Metro and the returned mesh has its uri replaced.
 */
export async function prepareMeshForViro(mesh: MeshInfo): Promise<MeshInfo> {
  if (mesh.isBundled) return mesh;
  if (mesh.uri.startsWith('http://') || mesh.uri.startsWith('https://')) return mesh;
  if (mesh.format !== 'GLB') return mesh;
  const httpUri = await uploadMeshToMetro(mesh.uri, mesh.id);
  return { ...mesh, uri: httpUri };
}

const BUNDLED_MESHES: MeshInfo[] = [];

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
    if (ext !== 'GLB' && ext !== 'OBJ' && ext !== 'PLY') continue;

    const filePath = `${IMPORTED_MESH_DIR}${file}`;
    const info = await FileSystem.getInfoAsync(filePath);
    const name = file.replace(/\.(glb|obj|ply)$/i, '');

    const fileUri = filePath.startsWith('file://') ? filePath : `file://${filePath}`;

    meshes.push({
      id: `imported_${file}`,
      name,
      format: ext as 'GLB' | 'OBJ' | 'PLY',
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

  if (ext !== 'GLB' && ext !== 'OBJ' && ext !== 'PLY') {
    return null;
  }

  if (!asset.uri) return null;

  await FileSystem.makeDirectoryAsync(IMPORTED_MESH_DIR, { intermediates: true });

  const name = fileName.replace(/\.(glb|obj|ply)$/i, '');

  if (ext === 'PLY') {
    // Convert PLY → GLB on-device so it can be rendered in AR
    const tempPlyPath = `${IMPORTED_MESH_DIR}_tmp_${Date.now()}.ply`;
    await FileSystem.copyAsync({ from: asset.uri, to: tempPlyPath });
    const glbFileName = `${name}.glb`;
    const glbPath = `${IMPORTED_MESH_DIR}${glbFileName}`;
    await convertPlyToGlb(tempPlyPath, glbPath);
    await FileSystem.deleteAsync(tempPlyPath, { idempotent: true });
    const info = await FileSystem.getInfoAsync(glbPath);
    return {
      id: `imported_${glbFileName}`,
      name,
      format: 'GLB',
      size: (info as any).size ?? 0,
      uri: glbPath,
      isBundled: false,
    };
  }

  const destPath = `${IMPORTED_MESH_DIR}${fileName}`;
  await FileSystem.copyAsync({
    from: asset.uri,
    to: destPath,
  });

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
    throw new Error('Bundled meshes are no longer supported. Please import a mesh.');
  }
  // http(s):// URIs (Metro-served) are passed through as-is.
  // Local paths get file:// prefix as a fallback.
  let uri = mesh.uri;
  if (!uri.startsWith('http://') && !uri.startsWith('https://') && !uri.startsWith('file://')) {
    uri = `file://${uri}`;
  }
  return {
    source: { uri },
    type: mesh.format,
  };
}

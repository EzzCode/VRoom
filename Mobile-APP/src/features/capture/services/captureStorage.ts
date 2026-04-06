import * as FileSystem from 'expo-file-system/legacy';

function getDocumentDirectory(): string {
  const documentDirectory = (FileSystem as { documentDirectory?: string }).documentDirectory;
  if (!documentDirectory) {
    throw new Error('Unable to resolve document directory for capture storage.');
  }
  return documentDirectory;
}

function toFileUri(path: string): string {
  if (path.startsWith('file://')) {
    return path;
  }
  return `file://${path}`;
}

export async function saveCapturedPhoto(photoPath: string, filenamePrefix = 'pose'): Promise<string> {
  const capturesDirectory = `${getDocumentDirectory()}captures/`;

  await FileSystem.makeDirectoryAsync(capturesDirectory, { intermediates: true });

  const filename = `${filenamePrefix}_${Date.now()}.jpg`;
  const destinationPath = `${capturesDirectory}${filename}`;
  const sourcePath = toFileUri(photoPath);

  try {
    await FileSystem.moveAsync({
      from: sourcePath,
      to: destinationPath,
    });
  } catch {
    // Some Android cache files are not movable across providers; copy then remove source.
    await FileSystem.copyAsync({
      from: sourcePath,
      to: destinationPath,
    });
    await FileSystem.deleteAsync(sourcePath, { idempotent: true });
  }

  return destinationPath;
}

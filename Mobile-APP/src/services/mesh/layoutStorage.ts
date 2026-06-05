import * as FileSystem from 'expo-file-system/legacy';
import { MeshInfo } from '../../shared/core/types';

const LAYOUTS_FILE_PATH = `${FileSystem.documentDirectory}room_layouts.json`;

export interface LayoutMesh {
  meshInfo: MeshInfo;
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
}

export interface RoomLayout {
  id: string;
  name: string;
  meshes: LayoutMesh[];
  createdAt: number;
  screenshotUri?: string; // URI of the ghost image used for alignment
}

/**
 * Loads all saved room layouts from the file system.
 */
export async function loadLayouts(): Promise<RoomLayout[]> {
  try {
    const info = await FileSystem.getInfoAsync(LAYOUTS_FILE_PATH);
    if (!info.exists) {
      return [];
    }
    const content = await FileSystem.readAsStringAsync(LAYOUTS_FILE_PATH);
    const layouts = JSON.parse(content) as RoomLayout[];
    // Sort by newest first
    return layouts.sort((a, b) => b.createdAt - a.createdAt);
  } catch (e) {
    console.error('Failed to load room layouts:', e);
    return [];
  }
}

/**
 * Saves a new room layout or updates an existing one.
 */
export async function saveLayout(layout: RoomLayout): Promise<void> {
  try {
    const existingLayouts = await loadLayouts();
    const index = existingLayouts.findIndex((l) => l.id === layout.id);
    
    if (index >= 0) {
      existingLayouts[index] = layout;
    } else {
      existingLayouts.push(layout);
    }
    
    await FileSystem.writeAsStringAsync(
      LAYOUTS_FILE_PATH,
      JSON.stringify(existingLayouts, null, 2)
    );
  } catch (e) {
    console.error('Failed to save room layout:', e);
    throw e;
  }
}

/**
 * Deletes a room layout by its ID.
 */
export async function deleteLayout(id: string): Promise<void> {
  try {
    const existingLayouts = await loadLayouts();
    const newLayouts = existingLayouts.filter((l) => l.id !== id);
    
    await FileSystem.writeAsStringAsync(
      LAYOUTS_FILE_PATH,
      JSON.stringify(newLayouts, null, 2)
    );
  } catch (e) {
    console.error('Failed to delete room layout:', e);
    throw e;
  }
}

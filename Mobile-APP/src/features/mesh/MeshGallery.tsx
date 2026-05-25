import React, { useCallback, useEffect, useState } from 'react';
import { View, FlatList, Text, StyleSheet, ActivityIndicator, Alert, TouchableOpacity } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useTheme } from '../../shared/theme';
import { Header, MeshCard, Button } from '../../shared/components';
import {
  getAvailableMeshes,
  importMeshFromFilePicker,
  deleteImportedMesh,
  formatFileSize,
} from '../../services/mesh/meshStorage';
import { MeshInfo } from '../../shared/core/types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'MeshGallery'>;

export default function MeshGallery({ navigation }: Props) {
  const { theme } = useTheme();
  const [meshes, setMeshes] = useState<MeshInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const loadMeshes = useCallback(async () => {
    setLoading(true);
    try {
      const available = await getAvailableMeshes();
      setMeshes(available);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMeshes();
  }, [loadMeshes]);

  const handleImport = useCallback(async () => {
    const mesh = await importMeshFromFilePicker();
    if (mesh) {
      setMeshes((prev) => [...prev, mesh]);
    }
  }, []);

  const handleDelete = useCallback((mesh: MeshInfo) => {
    Alert.alert(
      'Delete Mesh',
      `Delete "${mesh.name}"? This cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            await deleteImportedMesh(mesh);
            setMeshes((prev) => prev.filter((m) => m.id !== mesh.id));
          },
        },
      ],
    );
  }, []);

  const handleMeshPress = useCallback(
    (mesh: MeshInfo) => {
      if (mesh.format === 'PLY') {
        Alert.alert(
          'Format Not Supported in AR',
          'PLY files cannot be viewed in AR. Please convert your file to GLB or OBJ first.',
          [{ text: 'OK' }],
        );
        return;
      }
      navigation.navigate('ARView', {
        meshId: mesh.id,
        meshName: mesh.name,
        meshUri: mesh.uri,
        meshType: mesh.format,
        isBundled: mesh.isBundled,
      });
    },
    [navigation],
  );

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Header title="Mesh Library" onBack={() => navigation.goBack()} />

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={theme.colors.primary} size="large" />
        </View>
      ) : meshes.length === 0 ? (
        <View style={styles.center}>
          <Text
            style={{
              color: theme.colors.textTertiary,
              fontSize: theme.typography.body.fontSize,
              textAlign: 'center',
            }}
          >
            No meshes yet.{'\n'}Import a .glb or .obj file to get started.
          </Text>
        </View>
      ) : (
        <FlatList
          data={meshes}
          keyExtractor={(item) => item.id}
          numColumns={2}
          columnWrapperStyle={styles.row}
          contentContainerStyle={{ padding: theme.spacing.lg, gap: theme.spacing.md }}
          renderItem={({ item }) => (
            <View style={styles.gridItem}>
              <MeshCard
                name={item.name}
                format={item.format}
                size={formatFileSize(item.size)}
                onPress={() => handleMeshPress(item)}
              />
              {!item.isBundled && (
                <TouchableOpacity
                  style={styles.deleteBtn}
                  onPress={() => handleDelete(item)}
                  hitSlop={{ top: 6, right: 6, bottom: 6, left: 6 }}
                >
                  <Ionicons name="close-circle" size={22} color="#e53935" />
                </TouchableOpacity>
              )}
            </View>
          )}
        />
      )}

      <View style={[styles.fab, { bottom: theme.spacing.xl, right: theme.spacing.xl }]}>
        <Button title="Import" onPress={handleImport} variant="primary" size="md" />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 40,
  },
  row: {
    gap: 12,
  },
  gridItem: {
    flex: 1,
    maxWidth: '50%',
    position: 'relative',
  },
  deleteBtn: {
    position: 'absolute',
    top: -8,
    right: -8,
    zIndex: 10,
    backgroundColor: 'white',
    borderRadius: 11,
  },
  fab: {
    position: 'absolute',
  },
});

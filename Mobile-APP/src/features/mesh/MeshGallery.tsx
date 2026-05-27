import React, { useCallback, useEffect, useState } from 'react';
import { View, FlatList, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { useTheme } from '../../shared/theme';
import { Header, MeshCard, Button } from '../../shared/components';
import {
  getAvailableMeshes,
  importMeshFromFilePicker,
  formatFileSize,
  prepareMeshForViro,
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

  const handleMeshPress = useCallback(
    async (mesh: MeshInfo) => {
      let prepared = mesh;
      try {
        prepared = await prepareMeshForViro(mesh);
      } catch (e) {
        console.warn('Mesh upload to Metro failed, falling back to local URI:', e);
      }
      navigation.navigate('ARView', {
        meshId: prepared.id,
        meshName: prepared.name,
        meshUri: prepared.uri,
        meshType: prepared.format,
        isBundled: prepared.isBundled,
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
  },
  fab: {
    position: 'absolute',
  },
});

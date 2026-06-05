import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, ActivityIndicator, SafeAreaView } from 'react-native';
import { useTheme } from '../../shared/theme';
import { IconButton } from '../../shared/components';
import { RoomLayout, loadLayouts, deleteLayout } from '../../services/mesh/layoutStorage';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'SavedLayouts'>;

export default function SavedLayoutsScreen({ navigation }: Props) {
  const { theme } = useTheme();
  const [layouts, setLayouts] = useState<RoomLayout[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchLayouts = async () => {
    setIsLoading(true);
    const loaded = await loadLayouts();
    setLayouts(loaded);
    setIsLoading(false);
  };

  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      fetchLayouts();
    });
    return unsubscribe;
  }, [navigation]);

  const handleDelete = async (id: string) => {
    await deleteLayout(id);
    fetchLayouts();
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <View style={styles.header}>
        <IconButton icon="arrow-back" onPress={() => navigation.goBack()} />
        <Text
          style={{
            color: theme.colors.textPrimary,
            fontSize: theme.typography.h4.fontSize,
            fontWeight: '700',
            marginLeft: 12,
          }}
        >
          Saved Layouts
        </Text>
      </View>

      {isLoading ? (
        <ActivityIndicator style={{ flex: 1 }} color={theme.colors.primary} size="large" />
      ) : layouts.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={{ fontSize: 64, marginBottom: 16 }}>🛋️</Text>
          <Text style={{ color: theme.colors.textPrimary, fontSize: 20, fontWeight: '600' }}>
            No Layouts Saved
          </Text>
          <Text
            style={{
              color: theme.colors.textSecondary,
              textAlign: 'center',
              marginTop: 8,
              paddingHorizontal: 32,
            }}
          >
            Create your first layout by projecting an object in AR and tapping "Save Current" in the Layouts menu.
          </Text>
        </View>
      ) : (
        <FlatList
          data={layouts}
          keyExtractor={(item) => item.id}
          contentContainerStyle={{ padding: theme.spacing.lg }}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={[
                styles.card,
                { backgroundColor: theme.colors.card, borderColor: theme.colors.border },
              ]}
              onPress={() => navigation.navigate('ARView', { layout: item })}
            >
              <View style={{ flex: 1 }}>
                <Text
                  style={{
                    color: theme.colors.textPrimary,
                    fontSize: theme.typography.h4.fontSize,
                    fontWeight: '600',
                  }}
                  numberOfLines={1}
                >
                  {item.name}
                </Text>
                <Text
                  style={{
                    color: theme.colors.textSecondary,
                    fontSize: theme.typography.body.fontSize,
                    marginTop: 4,
                  }}
                >
                  {item.meshes.length} object{item.meshes.length !== 1 ? 's' : ''} •{' '}
                  {new Date(item.createdAt).toLocaleDateString()}
                </Text>
              </View>
              <IconButton
                icon="trash-outline"
                color={theme.colors.error}
                onPress={() => handleDelete(item.id)}
              />
            </TouchableOpacity>
          )}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 16,
    borderWidth: 1,
    marginBottom: 12,
  },
});

import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, FlatList, Image, TouchableOpacity, Alert, SafeAreaView, Dimensions } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import { useTheme } from '../../shared/theme';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'CapturedPhotos'>;

export default function CapturedPhotosScreen({ navigation }: Props) {
  const { theme } = useTheme();
  const [photos, setPhotos] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const loadPhotos = async () => {
    try {
      const documentDirectory = (FileSystem as { documentDirectory?: string }).documentDirectory;
      if (!documentDirectory) return;
      
      const capturesDir = `${documentDirectory}captures/`;
      const dirInfo = await FileSystem.getInfoAsync(capturesDir);
      
      if (dirInfo.exists) {
        const files = await FileSystem.readDirectoryAsync(capturesDir);
        const imageUris = files
          .filter(f => f.endsWith('.jpg') || f.endsWith('.png'))
          .map(f => `${capturesDir}${f}`)
          .sort()
          .reverse(); // Newest first
        setPhotos(imageUris);
      }
    } catch (e) {
      console.warn('Error reading captures:', e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadPhotos();
  }, []);

  const handleDeleteAll = async () => {
    Alert.alert(
      'Delete All',
      'Are you sure you want to delete all captured photos?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: async () => {
            try {
              const documentDirectory = (FileSystem as { documentDirectory?: string }).documentDirectory;
              if (documentDirectory) {
                await FileSystem.deleteAsync(`${documentDirectory}captures/`, { idempotent: true });
              }
              setPhotos([]);
            } catch (e) {
              Alert.alert('Error', 'Failed to delete photos.');
            }
          }
        }
      ]
    );
  };

  const renderItem = ({ item }: { item: string }) => {
    const size = Dimensions.get('window').width / 3 - 4; // 3 columns
    return (
      <View style={{ margin: 2 }}>
        <Image source={{ uri: item }} style={{ width: size, height: size, borderRadius: 4 }} />
      </View>
    );
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <View style={[styles.header, { borderBottomColor: theme.colors.border }]}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={{ padding: 8 }}>
          <Text style={{ color: theme.colors.primary, fontSize: 16 }}>← Back</Text>
        </TouchableOpacity>
        <Text style={{ color: theme.colors.textPrimary, fontSize: 18, fontWeight: 'bold' }}>
          Capture Gallery
        </Text>
        <TouchableOpacity onPress={handleDeleteAll} style={{ padding: 8 }} disabled={photos.length === 0}>
          <Text style={{ color: photos.length ? '#ff4444' : theme.colors.textTertiary, fontSize: 16 }}>Clear</Text>
        </TouchableOpacity>
      </View>

      {isLoading ? (
        <View style={styles.center}>
          <Text style={{ color: theme.colors.textSecondary }}>Loading...</Text>
        </View>
      ) : photos.length === 0 ? (
        <View style={styles.center}>
          <Text style={{ color: theme.colors.textSecondary }}>No photos captured yet.</Text>
          <Text style={{ color: theme.colors.textTertiary, marginTop: 8 }}>Start a scan to see images here.</Text>
        </View>
      ) : (
        <FlatList
          data={photos}
          keyExtractor={(item) => item}
          renderItem={renderItem}
          numColumns={3}
          contentContainerStyle={{ paddingVertical: 8 }}
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
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

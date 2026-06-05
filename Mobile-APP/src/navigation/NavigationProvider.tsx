import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useTheme } from '../shared/theme';
import HomeScreen from '../features/home/HomeScreen';
import CaptureScreen from '../features/capture/CaptureScreen';
import MeshGallery from '../features/mesh/MeshGallery';
import ARViewScreen from '../features/ar/ARViewScreen';
import ExportScreen from '../features/export/ExportScreen';
import ReconstructionStatusScreen from '../features/export/ReconstructionStatusScreen';
import CoverageDemoScreen from '../features/coverage/CoverageDemoScreen';
import { RootStackParamList } from './types';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function NavigationProvider() {
  const { theme } = useTheme();

  return (
    <NavigationContainer
      theme={{
        dark: true,
        colors: {
          primary: theme.colors.primary,
          background: theme.colors.background,
          card: theme.colors.card,
          text: theme.colors.textPrimary,
          border: theme.colors.border,
          notification: theme.colors.error,
        },
        fonts: {
          regular: { fontFamily: '', fontWeight: '400' },
          medium: { fontFamily: '', fontWeight: '500' },
          bold: { fontFamily: '', fontWeight: '700' },
          heavy: { fontFamily: '', fontWeight: '900' },
        },
      }}
    >
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: theme.colors.background },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Capture" component={CaptureScreen} />
        <Stack.Screen name="MeshGallery" component={MeshGallery} />
        <Stack.Screen
          name="ARView"
          component={ARViewScreen}
          options={{ animation: 'slide_from_bottom', orientation: 'portrait' }}
        />
        <Stack.Screen name="Export" component={ExportScreen} />
        <Stack.Screen
          name="ReconstructionStatus"
          component={ReconstructionStatusScreen}
          options={{ gestureEnabled: false }}
        />
        <Stack.Screen
          name="CoverageDemo"
          component={CoverageDemoScreen}
          options={{ animation: 'slide_from_bottom', orientation: 'portrait' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

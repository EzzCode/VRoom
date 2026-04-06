import React from 'react';
import { View, StatusBar, StyleSheet } from 'react-native';
import { SessionProvider } from './src/providers/SessionProvider';
import CaptureScreen from './src/features/capture/CaptureScreen';

export default function App() {
  return (
    <SessionProvider>
      <View style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="black" translucent />
        <CaptureScreen />
      </View>
    </SessionProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
});

import React from 'react';
import { StatusBar } from 'react-native';
import { ThemeProvider } from './src/shared/theme';
import NavigationProvider from './src/navigation/NavigationProvider';

export default function App() {
  return (
    <ThemeProvider>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
      <NavigationProvider />
    </ThemeProvider>
  );
}

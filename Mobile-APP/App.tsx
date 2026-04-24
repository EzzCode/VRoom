import React from 'react';
import { StatusBar } from 'react-native';
import { ThemeProvider } from './src/shared/theme';
import NavigationProvider from './src/navigation/NavigationProvider';
import { SessionProvider } from './src/providers/SessionProvider';

export default function App() {
  return (
    <ThemeProvider>
      <SessionProvider>
        <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
        <NavigationProvider />
      </SessionProvider>
    </ThemeProvider>
  );
}

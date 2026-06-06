import React from 'react';
import { StatusBar } from 'react-native';
import { ThemeProvider } from './src/shared/theme';
import { SessionProvider } from './src/providers/SessionProvider';
import NavigationProvider from './src/navigation/NavigationProvider';

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

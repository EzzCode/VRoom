import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { StyleSheet } from 'react-native';
import { darkTheme, lightTheme, ThemeTokens } from './tokens';

type ThemeMode = 'dark' | 'light';

interface ThemeContextValue {
  theme: ThemeTokens;
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
  toggleMode: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>('dark');

  const toggleMode = useCallback(() => {
    setMode((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  const theme = mode === 'dark' ? darkTheme : lightTheme;

  const value = useMemo(() => ({ theme, mode, setMode, toggleMode }), [theme, mode, toggleMode]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error('useTheme must be used within a <ThemeProvider>');
  }
  return ctx;
}

export function createStyles<T extends StyleSheet.NamedStyles<T>>(
  factory: (theme: ThemeTokens) => T,
): (theme: ThemeTokens) => T {
  return factory;
}

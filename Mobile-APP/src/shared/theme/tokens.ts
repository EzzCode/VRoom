type ShadowStyle = {
  shadowColor: string;
  shadowOffset: { width: number; height: number };
  shadowOpacity: number;
  shadowRadius: number;
  elevation: number;
};

export const colors = {
  primary: '#6C5CE7',
  primaryDark: '#5A4BD1',
  primaryLight: '#A29BFE',
  onPrimary: '#FFFFFF',
  secondary: '#00CEC9',
  secondaryDark: '#00B5B1',
  onSecondary: '#FFFFFF',
  surface: '#1E1E2E',
  surfaceLight: '#2D2D44',
  onSurface: '#FFFFFF',
  onSurfaceMuted: 'rgba(255, 255, 255, 0.6)',
  background: '#0F0F1A',
  backgroundElevated: '#181828',
  error: '#FF6B6B',
  errorBackground: 'rgba(255, 107, 107, 0.15)',
  success: '#00E676',
  successBackground: 'rgba(0, 230, 118, 0.15)',
  warning: '#FFD93D',
  warningBackground: 'rgba(255, 217, 61, 0.15)',
  recording: '#FF3B30',
  recordingBackground: 'rgba(255, 59, 48, 0.15)',
  border: 'rgba(255, 255, 255, 0.08)',
  borderLight: 'rgba(255, 255, 255, 0.12)',
  overlay: 'rgba(0, 0, 0, 0.6)',
  overlayLight: 'rgba(0, 0, 0, 0.3)',
  card: '#1A1A2E',
  cardElevated: '#22223A',
  textPrimary: '#FFFFFF',
  textSecondary: 'rgba(255, 255, 255, 0.7)',
  textTertiary: 'rgba(255, 255, 255, 0.4)',
  trackingNormal: '#00E676',
  trackingLimited: '#FFD93D',
  trackingUnavailable: '#FF6B6B',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
  xxxl: 48,
} as const;

export const typography = {
  h1: { fontSize: 32, fontWeight: '700' as const, lineHeight: 40 },
  h2: { fontSize: 24, fontWeight: '700' as const, lineHeight: 32 },
  h3: { fontSize: 20, fontWeight: '600' as const, lineHeight: 28 },
  h4: { fontSize: 17, fontWeight: '600' as const, lineHeight: 24 },
  body: { fontSize: 16, fontWeight: '400' as const, lineHeight: 22 },
  bodyBold: { fontSize: 16, fontWeight: '600' as const, lineHeight: 22 },
  caption: { fontSize: 13, fontWeight: '400' as const, lineHeight: 18 },
  captionBold: { fontSize: 13, fontWeight: '600' as const, lineHeight: 18 },
  mono: { fontSize: 14, fontWeight: '500' as const, lineHeight: 20 },
} as const;

export const radii = {
  sm: 6,
  md: 12,
  lg: 16,
  xl: 24,
  full: 9999,
} as const;

export const shadows: Record<string, ShadowStyle> = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 4,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.35,
    shadowRadius: 16,
    elevation: 8,
  },
};

export const animation = {
  fast: 150,
  normal: 250,
  slow: 400,
} as const;

export type ThemeTokens = {
  colors: typeof colors;
  spacing: typeof spacing;
  typography: typeof typography;
  radii: typeof radii;
  shadows: typeof shadows;
  animation: typeof animation;
};

export const lightTheme: ThemeTokens = {
  colors: {
    ...colors,
    surface: '#FFFFFF',
    surfaceLight: '#F5F5F7',
    onSurface: '#1A1A2E',
    onSurfaceMuted: 'rgba(0, 0, 0, 0.5)',
    background: '#F2F2F7',
    backgroundElevated: '#FFFFFF',
    border: 'rgba(0, 0, 0, 0.08)',
    borderLight: 'rgba(0, 0, 0, 0.12)',
    overlay: 'rgba(255, 255, 255, 0.6)',
    overlayLight: 'rgba(255, 255, 255, 0.3)',
    card: '#FFFFFF',
    cardElevated: '#FFFFFF',
    textPrimary: '#1A1A2E',
    textSecondary: 'rgba(0, 0, 0, 0.6)',
    textTertiary: 'rgba(0, 0, 0, 0.35)',
  },
  spacing,
  typography,
  radii,
  shadows: {
    sm: { ...shadows.sm, shadowOpacity: 0.08 } as ShadowStyle,
    md: { ...shadows.md, shadowOpacity: 0.12 } as ShadowStyle,
    lg: { ...shadows.lg, shadowOpacity: 0.18 } as ShadowStyle,
  },
  animation,
};

export const darkTheme: ThemeTokens = {
  colors,
  spacing,
  typography,
  radii,
  shadows,
  animation,
};

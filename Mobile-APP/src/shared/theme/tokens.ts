type ShadowStyle = {
  shadowColor: string;
  shadowOffset: { width: number; height: number };
  shadowOpacity: number;
  shadowRadius: number;
  elevation: number;
};

export const colors = {
  primary: '#22D3EE',
  primaryDark: '#0891B2',
  primaryLight: '#67E8F9',
  onPrimary: '#FFFFFF',
  secondary: '#60A5FA',
  secondaryDark: '#2563EB',
  onSecondary: '#FFFFFF',
  surface: '#17212B',
  surfaceLight: '#22313F',
  onSurface: '#FFFFFF',
  onSurfaceMuted: 'rgba(230, 242, 255, 0.62)',
  
  background: '#ffffffff',

  backgroundElevated: '#0F1B26',
  error: '#FB7185',
  errorBackground: 'rgba(251, 113, 133, 0.16)',
  success: '#34D399',
  successBackground: 'rgba(52, 211, 153, 0.16)',
  warning: '#FBBF24',
  warningBackground: 'rgba(251, 191, 36, 0.16)',
  recording: '#F43F5E',
  recordingBackground: 'rgba(244, 63, 94, 0.16)',
  border: 'rgba(148, 163, 184, 0.16)',
  borderLight: 'rgba(148, 163, 184, 0.24)',
  overlay: 'rgba(7, 16, 24, 0.72)',
  overlayLight: 'rgba(7, 16, 24, 0.42)',
  
  card: '#3a3a3a81',
  
  cardElevated: '#172635',
  textPrimary: '#FFFFFF',
  textSecondary: 'rgba(3, 3, 3, 1)',
  textTertiary: 'rgba(255, 255, 255, 0.48)',
  trackingNormal: '#34D399',
  trackingLimited: '#FBBF24',
  trackingUnavailable: '#FB7185',
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
    surfaceLight: '#EAF4F8',
    onSurface: '#10202B',
    onSurfaceMuted: 'rgba(16, 32, 43, 0.58)',
    background: '#F3F8FA',
    backgroundElevated: '#FFFFFF',
    border: 'rgba(15, 23, 42, 0.10)',
    borderLight: 'rgba(15, 23, 42, 0.16)',
    overlay: 'rgba(255, 255, 255, 0.72)',
    overlayLight: 'rgba(255, 255, 255, 0.42)',
    card: '#FFFFFF',
    cardElevated: '#F7FBFC',
    textPrimary: '#10202B',
    textSecondary: 'rgba(16, 32, 43, 0.68)',
    textTertiary: 'rgba(16, 32, 43, 0.44)',
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

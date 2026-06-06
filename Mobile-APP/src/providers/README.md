# Providers

React context providers holding app-level shared state.

| File | Purpose |
|---|---|
| `SessionProvider.tsx` | Owns the capture session: recording state, keyframes, the shared `CoverageTracker`, the `KeyframeExtractor` (with `AngleGate` + `CoverageGate` registered), current pose, and `getMetadata()` for export. Exposes the `useSession()` hook. |
| `ARProvider.tsx` | Owns AR mesh-placement state (selected mesh, interaction mode, tracking state, placement transform). Exposes the `useAR()` hook. |

Navigation lives separately in [src/navigation](../navigation), and the theme provider in [src/shared/theme](../shared/theme). `App.tsx` composes these providers and stays minimal.

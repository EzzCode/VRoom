# Services Layer

Cross-feature service modules shared by at least two features.

| Folder | State |
|---|---|
| `mesh/` | Active: `meshStorage.ts` (mesh file listing + `formatFileSize`), `layoutStorage.ts` (saved room layouts), `plyToGlb.ts` (PLY → GLB conversion). |
| `opencv/` | Placeholder — empty. |
| `storage/` | Placeholder — empty. |

Feature-local services should remain inside the owning feature folder until reuse is proven.

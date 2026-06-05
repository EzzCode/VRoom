import { type NativeStackScreenProps } from '@react-navigation/native-stack';

import { RoomLayout } from '../services/mesh/layoutStorage';

export type RootStackParamList = {
  Home: undefined;
  Capture: undefined;
  MeshGallery: undefined;
  SavedLayouts: undefined;
  ARView: {
    meshId?: string;
    meshName?: string;
    meshUri?: string;
    meshType?: 'GLB' | 'OBJ';
    isBundled?: boolean;
    layout?: RoomLayout;
  };
  Export: undefined;
  ReconstructionStatus: { jobId: string };
  CoverageDemo: undefined;
};

export type RootStackNavigation = NativeStackScreenProps<RootStackParamList>['navigation'];
export type RootStackRoute<T extends keyof RootStackParamList> = NativeStackScreenProps<
  RootStackParamList,
  T
>['route'];

declare global {
  namespace ReactNavigation {
    // eslint-disable-next-line @typescript-eslint/no-empty-object-type
    interface RootParamList extends RootStackParamList {}
  }
}

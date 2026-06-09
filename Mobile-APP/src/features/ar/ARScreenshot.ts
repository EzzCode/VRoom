import { Alert } from 'react-native';

export async function takeARScreenshot(arNavigatorRef: React.RefObject<any>): Promise<boolean> {
  try {
    const result = await arNavigatorRef.current?._takeScreenshot('vroom_ar', true);
    if (result?.success) {
      Alert.alert('Screenshot saved!', 'Your AR view has been saved to your photos.');
      return true;
    }
    return false;
  } catch (e) {
    console.warn('Screenshot failed:', e);
    return false;
  }
}

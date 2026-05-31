// Backend URL for reconstruction jobs.
// Override at runtime by editing this constant or wiring app settings.
// For local dev with a phone over USB, run on the laptop:
//   adb reverse tcp:8000 tcp:8000
// then keep the loopback URL below.
export const API_BASE_URL = 'http://127.0.0.1:8000';

export const POLL_INTERVAL_MS = 2500;

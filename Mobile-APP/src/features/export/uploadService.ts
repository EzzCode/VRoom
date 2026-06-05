import { Keyframe } from '../../shared/core/types';
import { API_BASE_URL, POLL_INTERVAL_MS } from './config';
import { SessionManifest } from './sessionPackager';

export type JobState =
  | 'queued'
  | 'colmap'
  | 'gaussian-training'
  | 'mesh-extraction'
  | 'done'
  | 'failed';

export interface JobStatus {
  jobId: string;
  state: JobState;
  /** 0..1 progress within current stage (optional) */
  progress?: number;
  message?: string;
  /** Filled when state === 'done' */
  resultUrl?: string;
}

export interface UploadResult {
  sessionId: string;
  jobId: string;
}

/**
 * Multipart upload of session.json + every keyframe JPG. Reports progress
 * via the optional onProgress callback (0..1).
 *
 * We use XHR because fetch() in React Native does not expose upload progress.
 */
export function uploadSession(
  manifest: SessionManifest,
  keyframes: Keyframe[],
  onProgress?: (fraction: number) => void,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const formData = new FormData();

    // Inline manifest as a plain string field; server reads request.form['sessionJson'].
    formData.append('sessionJson', JSON.stringify(manifest));

    for (let i = 0; i < keyframes.length; i++) {
      const kf = keyframes[i];
      if (!kf) continue;
      const manifestEntry = manifest.keyframes[i];
      const filename = manifestEntry?.filename ?? `frame_${kf.index}.jpg`;
      const uri = kf.imagePath.startsWith('file://') ? kf.imagePath : `file://${kf.imagePath}`;
      formData.append('images', {
        uri,
        name: filename,
        type: 'image/jpeg',
      } as unknown as Blob);
    }

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/sessions`);
    xhr.setRequestHeader('Accept', 'application/json');

    xhr.upload.onprogress = (evt) => {
      if (onProgress && evt.lengthComputable) {
        onProgress(evt.loaded / evt.total);
      }
    };

    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(`Upload failed: ${xhr.status} ${xhr.responseText}`));
        return;
      }
      try {
        const data = JSON.parse(xhr.responseText) as UploadResult;
        if (!data.jobId) {
          reject(new Error('Server response missing jobId'));
          return;
        }
        resolve(data);
      } catch (e) {
        reject(new Error(`Bad server response: ${(e as Error).message}`));
      }
    };

    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.ontimeout = () => reject(new Error('Upload timed out'));

    xhr.send(formData);
  });
}

export async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const resp = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
  if (!resp.ok) {
    throw new Error(`Job status failed: ${resp.status}`);
  }
  return (await resp.json()) as JobStatus;
}

/**
 * Polls /jobs/:id until state is 'done' or 'failed'. Calls onUpdate for each
 * status received. Returns the final status.
 */
export async function pollJobUntilComplete(
  jobId: string,
  onUpdate: (status: JobStatus) => void,
  signal?: AbortSignal,
): Promise<JobStatus> {
  while (true) {
    if (signal?.aborted) {
      throw new Error('Polling aborted');
    }
    const status = await fetchJobStatus(jobId);
    onUpdate(status);
    if (status.state === 'done' || status.state === 'failed') {
      return status;
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
}

/** Returns the URL to download the reconstructed GLB once a job is done. */
export function getJobResultUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${jobId}/result`;
}

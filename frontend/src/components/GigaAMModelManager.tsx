import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Button } from './ui/button';
import { RawModelInfo } from '@/hooks/useTranscriptionModels';

const MODEL = 'gigaam-v3-e2e-rnnt-int8';
export function GigaAMModelManager({ selectedModel, onModelSelect }: {
  selectedModel?: string; onModelSelect: (name: string) => void;
}) {
  const [model, setModel] = useState<RawModelInfo>();
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const refresh = async () => {
    const models = await invoke<RawModelInfo[]>('gigaam_get_available_models');
    setModel(models[0]);
  };
  useEffect(() => {
    let disposed = false;
    const unlisteners: (() => void)[] = [];
    const register = async () => {
      const registrations = await Promise.all([
        listen<number>('gigaam-download-progress', e => { if (!disposed) setProgress(e.payload); }),
        listen('gigaam-download-finished', () => { if (!disposed) { setBusy(false); refresh().catch(e => setError(String(e))); } }),
      ]);
      if (disposed) registrations.forEach(f => f()); else unlisteners.push(...registrations);
      if (!disposed) await refresh();
    };
    register().catch(e => { if (!disposed) setError(String(e)); });
    return () => { disposed = true; unlisteners.forEach(f => f()); };
  }, []);
  const downloading = busy || !!(model?.status && typeof model.status === 'object' && 'Downloading' in model.status);
  const available = model?.status === 'Available';
  const download = async () => {
    setBusy(true); setError(''); setProgress(0);
    try { await invoke('gigaam_download_model', { modelName: MODEL }); }
    catch (e) { setError(String(e)); }
    finally { setBusy(false); await refresh().catch(e => setError(String(e))); }
  };
  const select = async () => {
    setBusy(true); setError('');
    try {
      await invoke('gigaam_load_model', { modelName: MODEL });
      await invoke('api_save_transcript_config', { provider: 'gigaam', model: MODEL, apiKey: null });
      onModelSelect(MODEL);
    } catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  };
  return <div className="space-y-3 rounded-lg border p-4">
    <h3 className="font-medium">GigaAM v3 E2E RNNT · INT8</h3>
    <p className="text-sm text-gray-600">Russian · Local · macOS Apple Silicon · 227 MB download. Audio stays on this Mac.</p>
    {downloading && !available && <div role="status">Downloading: {progress}%<progress className="ml-2" value={progress} max={100} /></div>}
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
    {model?.status && typeof model.status === 'object' && 'Error' in model.status && <p role="alert">{model.status.Error}</p>}
    {available
      ? <Button disabled={busy} onClick={select}>{selectedModel === MODEL ? 'Selected — GigaAM' : 'Use GigaAM'}</Button>
      : downloading
        ? <Button variant="outline" onClick={() => invoke('gigaam_cancel_download').catch(e => setError(String(e)))}>Cancel download</Button>
        : <Button onClick={download}>{model?.status === 'Missing' ? 'Download model' : 'Retry download'}</Button>}
  </div>;
}

mod activity;
pub mod model;

use crate::audio::transcription::{TranscriptResult, TranscriptionError, TranscriptionProvider};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio::{io::AsyncWriteExt, sync::Mutex};

pub const MODEL: &str = "gigaam-v3-e2e-rnnt-int8";
pub static ENGINE: Lazy<Arc<GigaProvider>> = Lazy::new(|| {
    Arc::new(GigaProvider {
        model: Arc::new(std::sync::Mutex::new(None)),
    })
});
static DOWNLOAD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
static CANCEL: Lazy<std::sync::Mutex<tokio_util::sync::CancellationToken>> =
    Lazy::new(|| std::sync::Mutex::new(tokio_util::sync::CancellationToken::new()));
static PROGRESS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub struct GigaProvider {
    model: Arc<std::sync::Mutex<Option<model::GigaModel>>>,
}

fn supported() -> Result<()> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        bail!("GigaAM currently supports macOS Apple Silicon only");
    }
    Ok(())
}
pub fn directory<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf> {
    Ok(app.path().app_data_dir()?.join("models").join(MODEL))
}
fn files() -> Vec<(String, String, u64, String)> {
    let lock: serde_json::Value = serde_json::from_str(include_str!("models.lock.json")).unwrap();
    let mut files: Vec<_> = lock["files"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, meta)| {
            (
                name.clone(),
                format!(
                    "https://huggingface.co/{}/resolve/{}/{}",
                    lock["repo"].as_str().unwrap(),
                    lock["revision"].as_str().unwrap(),
                    name
                ),
                meta["bytes"].as_u64().unwrap(),
                meta["sha256"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let pre = &lock["preprocessor"];
    files.push((
        "gigaam_v3.onnx".into(),
        pre["url"].as_str().unwrap().into(),
        pre["bytes"].as_u64().unwrap(),
        pre["sha256"].as_str().unwrap().into(),
    ));
    files
}
fn verify(path: &Path, size: u64, hash: &str) -> Result<()> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    if f.metadata()?.len() != size {
        bail!("Size mismatch: {}", path.display());
    }
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    if format!("{:x}", digest.finalize()) != hash {
        bail!("SHA256 mismatch: {}", path.display());
    }
    Ok(())
}
fn verify_all(dir: &Path) -> Result<()> {
    for (name, _, size, hash) in files() {
        verify(&dir.join(name), size, &hash)?;
    }
    Ok(())
}
pub async fn load<R: Runtime>(app: &AppHandle<R>, name: &str) -> Result<Arc<GigaProvider>> {
    supported()?;
    if name != MODEL {
        bail!("Unknown GigaAM model: {name}");
    }
    let dir = directory(app)?;
    let engine = ENGINE.clone();
    let target = engine.clone();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let mut guard = target
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("GigaAM model lock poisoned"))?;
        if guard.is_none() {
            verify_all(&dir).context(
                "Download GigaAM in Transcript Settings first (or retry a corrupted download)",
            )?;
            *guard = Some(model::GigaModel::load(&dir)?);
        }
        Ok(())
    })
    .await??;
    Ok(engine)
}
pub async fn unload() {
    let engine = ENGINE.clone();
    let _ = tokio::task::spawn_blocking(move || {
        if let Ok(mut guard) = engine.model.lock() {
            *guard = None;
        }
    })
    .await;
}
#[async_trait]
impl TranscriptionProvider for GigaProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> Result<TranscriptResult, TranscriptionError> {
        if let Some(lang) = language {
            if !matches!(lang.as_str(), "ru" | "auto" | "") {
                return Err(TranscriptionError::UnsupportedLanguage(lang));
            }
        }
        let model = self.model.clone();
        let text = tokio::task::spawn_blocking(move || {
            let mut guard = model
                .lock()
                .map_err(|_| TranscriptionError::EngineFailed("GigaAM lock poisoned".into()))?;
            guard
                .as_mut()
                .ok_or(TranscriptionError::ModelNotLoaded)?
                .transcribe(&audio)
                .map_err(|e| TranscriptionError::EngineFailed(e.to_string()))
        })
        .await
        .map_err(|e| TranscriptionError::EngineFailed(e.to_string()))??;
        Ok(TranscriptResult {
            text,
            confidence: None,
            is_partial: false,
        })
    }
    async fn is_model_loaded(&self) -> bool {
        self.model.lock().map(|m| m.is_some()).unwrap_or(false)
    }
    async fn get_current_model(&self) -> Option<String> {
        if self.is_model_loaded().await {
            Some(MODEL.into())
        } else {
            None
        }
    }
    fn provider_name(&self) -> &'static str {
        "GigaAM"
    }
}
#[derive(Serialize)]
pub struct ModelInfo {
    name: String,
    size_mb: u64,
    status: serde_json::Value,
}
#[tauri::command]
pub async fn gigaam_init() -> Result<(), String> {
    supported().map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn gigaam_get_available_models<R: Runtime>(
    app: AppHandle<R>,
) -> Result<Vec<ModelInfo>, String> {
    supported().map_err(|e| e.to_string())?;
    let dir = directory(&app).map_err(|e| e.to_string())?;
    let status = if DOWNLOAD.try_lock().is_err() {
        serde_json::json!({"Downloading": {"progress": PROGRESS.load(std::sync::atomic::Ordering::Relaxed)}})
    } else {
        tokio::task::spawn_blocking(move || {
            if verify_all(&dir).is_ok() {
                serde_json::json!("Available")
            } else if files()
                .iter()
                .any(|(name, _, _, _)| dir.join(name).exists())
            {
                serde_json::json!({"Error":"Incomplete or corrupted model. Retry download."})
            } else {
                serde_json::json!("Missing")
            }
        })
        .await
        .map_err(|e| e.to_string())?
    };
    Ok(vec![ModelInfo {
        name: MODEL.into(),
        size_mb: files().iter().map(|x| x.2).sum::<u64>() / 1_000_000,
        status,
    }])
}
#[tauri::command]
pub async fn gigaam_has_available_models<R: Runtime>(app: AppHandle<R>) -> Result<bool, String> {
    Ok(gigaam_get_available_models(app)
        .await?
        .iter()
        .any(|m| m.status == "Available"))
}
#[tauri::command]
pub async fn gigaam_load_model<R: Runtime>(
    app: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    load(&app, &model_name)
        .await
        .map(|_| ())
        .map_err(|e| format!("{e:#}"))
}
#[tauri::command]
pub async fn gigaam_cancel_download() -> Result<(), String> {
    CANCEL
        .lock()
        .map_err(|_| "Download state lock poisoned")?
        .cancel();
    Ok(())
}
#[tauri::command]
pub async fn gigaam_download_model<R: Runtime>(
    app: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    supported().map_err(|e| e.to_string())?;
    if model_name != MODEL {
        return Err("Unknown GigaAM model".into());
    }
    let _guard = DOWNLOAD
        .try_lock()
        .map_err(|_| "GigaAM download already running")?;
    let cancel = tokio_util::sync::CancellationToken::new();
    *CANCEL.lock().map_err(|_| "Download state lock poisoned")? = cancel.clone();
    PROGRESS.store(0, std::sync::atomic::Ordering::Relaxed);
    let result = download(&app, cancel).await;
    drop(_guard);
    let _ = app.emit(
        "gigaam-download-finished",
        serde_json::json!({"error":result.as_ref().err().map(|e| format!("{e:#}"))}),
    );
    result.map_err(|e| format!("{e:#}"))
}
async fn download<R: Runtime>(
    app: &AppHandle<R>,
    cancel: tokio_util::sync::CancellationToken,
) -> Result<()> {
    let dir = directory(app)?;
    tokio::fs::create_dir_all(&dir).await?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(1800))
        .build()?;
    let total: u64 = files().iter().map(|x| x.2).sum();
    let mut done = 0;
    for (name, url, size, hash) in files() {
        if cancel.is_cancelled() {
            bail!("Download cancelled");
        }
        let path = dir.join(&name);
        let check = path.clone();
        let checksum = hash.clone();
        if tokio::task::spawn_blocking(move || verify(&check, size, &checksum).is_ok()).await? {
            done += size;
            continue;
        }
        let temp = dir.join(format!("{name}.part"));
        let result = async {
            let mut response = tokio::select! {
                _ = cancel.cancelled() => { bail!("Download cancelled"); },
                response = tokio::time::timeout(std::time::Duration::from_secs(60), client.get(url).send()) => response.context("Timed out waiting for model download headers")??.error_for_status()?,
            };
            let mut output = tokio::fs::File::create(&temp).await?;
            let mut count = 0;
            loop {
                let chunk = tokio::select! {
                    _ = cancel.cancelled() => { bail!("Download cancelled"); },
                    chunk = tokio::time::timeout(std::time::Duration::from_secs(60), response.chunk()) => chunk.context("Model download stalled; retry to keep verified files")??,
                };
                let Some(chunk) = chunk else {
                    break;
                };
                count += chunk.len() as u64;
                if count > size {
                    bail!("Downloaded file exceeds locked size");
                }
                output.write_all(&chunk).await?;
                let progress = ((done + count) * 100 / total) as u32;
                if PROGRESS.swap(progress, std::sync::atomic::Ordering::Relaxed) != progress {
                    let _ = app.emit("gigaam-download-progress", progress);
                }
            }
            output.flush().await?;
            output.sync_all().await?;
            drop(output);
            let check = temp.clone();
            tokio::task::spawn_blocking(move || verify(&check, size, &hash)).await??;
            tokio::fs::rename(&temp, &path).await?;
            Ok::<_, anyhow::Error>(())
        }
        .await;
        if result.is_err() {
            let _ = tokio::fs::remove_file(&temp).await;
        }
        result?;
        done += size;
    }
    PROGRESS.store(100, std::sync::atomic::Ordering::Relaxed);
    let _ = app.emit("gigaam-download-progress", 100u32);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integrity_rejects_truncation_and_same_length_corruption() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model");
        let hash = format!("{:x}", Sha256::digest(b"correct"));
        std::fs::write(&path, b"correct")?;
        verify(&path, 7, &hash)?;
        std::fs::write(&path, b"corrupt")?;
        assert!(verify(&path, 7, &hash).is_err());
        std::fs::write(&path, b"short")?;
        assert!(verify(&path, 7, &hash).is_err());
        Ok(())
    }
    #[test]
    fn locked_assets_are_pinned_and_include_all_rnnt_components() {
        let files = files();
        assert_eq!(files.len(), 6);
        assert!(files
            .iter()
            .all(|(_, url, bytes, hash)| url.starts_with("https://")
                && *bytes > 0
                && hash.len() == 64));
        for component in ["encoder", "decoder", "joint", "vocab"] {
            assert!(files.iter().any(|(name, _, _, _)| name.contains(component)));
        }
    }
}

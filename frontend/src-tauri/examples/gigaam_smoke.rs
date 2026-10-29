//! Exercise actual Tauri/Rust paths locally without capturing microphone audio.
//! Usage: gigaam_smoke AUDIO MODEL_DIR REPORT_JSON [--download]
//! Uses a separate application data directory and creates a validation meeting
//! in the usual recordings folder. Private audio/transcripts never enter Git.
use anyhow::{Context, Result};
use app_lib::{audio, database::manager::DatabaseManager, gigaam_engine, state::AppState};
use serde_json::{json, Value};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use tauri::{Listener, Manager};

async fn check(app: tauri::AppHandle, args: Vec<String>) -> Result<Value> {
    let db = DatabaseManager::new_from_app_handle(&app).await?;
    app.manage(AppState {
        db_manager: db.clone(),
    });
    let dir = gigaam_engine::directory(&app)?;
    if args.iter().any(|x| x == "--download") {
        gigaam_engine::gigaam_download_model(app.clone(), gigaam_engine::MODEL.into())
            .await
            .map_err(anyhow::Error::msg)?;
    } else {
        std::fs::create_dir_all(&dir)?;
        for entry in std::fs::read_dir(&args[2])? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                std::fs::copy(entry.path(), dir.join(entry.file_name()))?;
            }
        }
    }
    anyhow::ensure!(
        gigaam_engine::gigaam_has_available_models(app.clone())
            .await
            .map_err(anyhow::Error::msg)?,
        "Model unavailable"
    );
    app_lib::api::api::api_save_transcript_config(
        app.clone(),
        app.state(),
        "gigaam".into(),
        gigaam_engine::MODEL.into(),
        None,
        None,
    )
    .await
    .map_err(anyhow::Error::msg)?;
    let saved = app_lib::api::api::api_get_transcript_config(app.clone(), app.state(), None)
        .await
        .map_err(anyhow::Error::msg)?
        .context("Missing saved transcript configuration")?;
    anyhow::ensure!(saved.provider == "gigaam", "Provider did not persist");
    audio::transcription::validate_transcription_model_ready(&app)
        .await
        .map_err(anyhow::Error::msg)?;

    // Feed a speech excerpt from the provided recording into the SAME worker
    // and event path used by a meeting's VAD pipeline, without recording devices.
    let decoded = audio::decoder::decode_audio_file(Path::new(&args[1]))?;
    let pcm = decoded.to_whisper_format();
    let excerpt = pcm
        .get(100 * 16000..120 * 16000)
        .context("Need a recording >=120 seconds")?
        .to_vec();
    drop(pcm);
    drop(decoded);
    let updates = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = updates.clone();
    let listener = app.listen("transcript-update", move |e| {
        if let Ok(value) = serde_json::from_str(e.payload()) {
            received.lock().unwrap().push(value);
        }
    });
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = audio::transcription::start_transcription_task(app.clone(), rx);
    tx.send(audio::recording_state::AudioChunk {
        data: excerpt,
        sample_rate: 16000,
        timestamp: 100.0,
        chunk_id: 1,
        device_type: audio::recording_state::DeviceType::System,
    })?;
    drop(tx);
    tokio::time::timeout(std::time::Duration::from_secs(120), worker).await??;
    app.unlisten(listener);
    let live_count = updates.lock().unwrap().len();
    anyhow::ensure!(live_count > 0, "Live worker emitted no transcripts");
    gigaam_engine::unload().await;
    eprintln!("Live worker/event path passed");

    let began = std::time::Instant::now();
    let imported = audio::import::start_import(
        app.clone(),
        args[1].clone(),
        "GigaAM validation".into(),
        None,
        Some(gigaam_engine::MODEL.into()),
        Some("gigaam".into()),
    )
    .await?;
    anyhow::ensure!(
        imported.segments_count > 0,
        "Import produced no transcripts"
    );
    let folder: String = sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id = ?")
        .bind(&imported.meeting_id)
        .fetch_one(db.pool())
        .await?;
    let original: Value =
        serde_json::from_slice(&std::fs::read(Path::new(&folder).join("transcripts.json"))?)?;
    let import_s = began.elapsed().as_secs_f64();
    eprintln!("Import passed: {} segments", imported.segments_count);
    let began = std::time::Instant::now();
    let repeated = audio::retranscription::start_retranscription(
        app.clone(),
        imported.meeting_id.clone(),
        folder.clone(),
        None,
        Some(gigaam_engine::MODEL.into()),
        Some("gigaam".into()),
    )
    .await?;
    let retranscribe_s = began.elapsed().as_secs_f64();
    anyhow::ensure!(
        repeated.segments_count > 0,
        "Retranscription produced no transcripts"
    );
    let updated: Value =
        serde_json::from_slice(&std::fs::read(Path::new(&folder).join("transcripts.json"))?)?;
    let texts = |v: &Value| -> Vec<String> {
        v["segments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["text"].as_str().unwrap().to_owned())
            .collect()
    };
    let identical = texts(&original) == texts(&updated);
    anyhow::ensure!(identical, "Import/retranscription text differs");
    eprintln!(
        "Retranscription passed: {} segments, identical text",
        repeated.segments_count
    );
    Ok(
        json!({"live_worker_updates":live_count,"import_segments":imported.segments_count,
        "retranscription_segments":repeated.segments_count,"identical_text":identical,
        "duration_s": imported.duration_seconds,"import_s":import_s,"retranscription_s":retranscribe_s,
        "meeting_id":imported.meeting_id,"meeting_folder":folder,"model_directory":dir}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 4,
        "Usage: gigaam_smoke AUDIO MODEL_DIR REPORT_JSON [--download]"
    );
    let mut context = tauri::generate_context!();
    context.config_mut().identifier = "com.alexanderleo0.meetily.validation".into();
    context.config_mut().app.windows.clear();
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .setup(move |app| {
            let handle = app.handle().clone();
            let args = args.clone();
            tauri::async_runtime::spawn(async move {
                let result = check(handle.clone(), args.clone()).await;
                let (value, code) = match result {
                    Ok(value) => (value, 0),
                    Err(error) => {
                        eprintln!("Validation failed: {error:#}");
                        (json!({"error":format!("{error:#}")}), 1)
                    }
                };
                if let Err(error) =
                    std::fs::write(&args[3], serde_json::to_vec_pretty(&value).unwrap())
                {
                    eprintln!("Report write failed: {error}");
                }
                handle.exit(code);
            });
            Ok(())
        })
        .run(context)?;
    Ok(())
}

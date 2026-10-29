//! Local native parity check against the Python prototype's shared VAD segments.
//! Arguments: MODEL_DIR PCM_F32LE MANIFEST_JSON OUTPUT_JSON
//! Writes private transcripts only to the explicitly requested output, never logs them.
use anyhow::{Context, Result};
use app_lib::gigaam_engine::model::GigaModel;
use serde_json::{json, Value};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 5,
        "Usage: gigaam_verify MODEL_DIR PCM_F32LE MANIFEST_JSON OUTPUT_JSON"
    );
    let data = std::fs::read(&args[2])?;
    anyhow::ensure!(data.len() % 4 == 0, "Expected f32 little endian PCM");
    let audio: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let manifest: Value = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    anyhow::ensure!(
        manifest["sample_rate"] == 16000,
        "Expected 16kHz mono audio"
    );
    let mut model = GigaModel::load(std::path::Path::new(&args[1]))?;
    let mut results = vec![];
    let began = std::time::Instant::now();
    for (i, segment) in manifest["segments"]
        .as_array()
        .context("Missing segments")?
        .iter()
        .enumerate()
    {
        let start = segment["start"].as_u64().context("Missing start")? as usize;
        let end = segment["end"].as_u64().context("Missing end")? as usize;
        let chunk = audio.get(start..end).context("Segment outside audio")?;
        let text = model.transcribe(chunk)?;
        results.push(json!({"start":start,"end":end,"text":text}));
        if (i + 1) % 20 == 0 {
            eprintln!("Processed {} segments", i + 1);
        }
    }
    let elapsed = began.elapsed().as_secs_f64();
    let count = results.len();
    std::fs::write(
        &args[4],
        serde_json::to_vec_pretty(&json!({"inference_s":elapsed,"segments":results}))?,
    )?;
    eprintln!("Completed {count} segments in {elapsed:.2}s");
    Ok(())
}

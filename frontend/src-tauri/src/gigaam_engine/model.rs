//! GigaAM v3 E2E RNN-T greedy decoder. CPU INT8, 16 kHz mono.
//! Algorithm follows onnx-asr 0.12.0 (MIT), models/gigaam.py and asr.py.
use anyhow::{bail, Context, Result};
use ndarray::{s, Array1, Array2, Array3, ArrayD, IxDyn};
use ort::{inputs, session::Session, value::TensorRef};
use std::path::Path;

pub struct GigaModel {
    preprocessor: Session,
    encoder: Session,
    decoder: Session,
    joint: Session,
    vocab: Vec<String>,
    blank: usize,
}

impl GigaModel {
    pub fn load(dir: &Path) -> Result<Self> {
        super::activity::run(|| Self::load_inner(dir))
    }

    fn load_inner(dir: &Path) -> Result<Self> {
        crate::ensure_onnx_runtime_available()?;
        let session = |name: &str| -> Result<Session> {
            Ok(Session::builder()?
                .with_intra_threads(4)?
                .with_inter_threads(1)?
                .with_execution_providers([
                    ort::execution_providers::CPUExecutionProvider::default().build(),
                ])?
                .commit_from_file(dir.join(name))?)
        };
        let mut vocab = Vec::new();
        let mut blank = None;
        for line in std::fs::read_to_string(dir.join("v3_e2e_rnnt_vocab.txt"))?.lines() {
            let (token, id) = line.rsplit_once(' ').context("Invalid GigaAM vocabulary")?;
            let id: usize = id.parse()?;
            if id > 65536 {
                bail!("Invalid vocabulary index");
            }
            vocab.resize(vocab.len().max(id + 1), String::new());
            vocab[id] = token.replace('\u{2581}', " ");
            if token == "<blk>" {
                blank = Some(id);
            }
        }
        Ok(Self {
            preprocessor: session("gigaam_v3.onnx")?,
            encoder: session("v3_e2e_rnnt_encoder.int8.onnx")?,
            decoder: session("v3_e2e_rnnt_decoder.int8.onnx")?,
            joint: session("v3_e2e_rnnt_joint.int8.onnx")?,
            blank: blank.context("Missing GigaAM blank token")?,
            vocab,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        super::activity::run(|| self.transcribe_inner(audio))
    }

    fn transcribe_inner(&mut self, audio: &[f32]) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        if audio.iter().any(|x| !x.is_finite()) {
            bail!("Non-finite audio samples");
        }
        // Bound memory and decoder work even if a caller bypasses VAD chunking.
        let mut parts = Vec::new();
        for chunk in audio.chunks(30 * 16000) {
            if chunk.len() < 1600 {
                continue;
            }
            parts.push(self.transcribe_chunk(chunk)?);
        }
        Ok(parts
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn transcribe_chunk(&mut self, audio: &[f32]) -> Result<String> {
        let wav = Array2::from_shape_vec((1, audio.len()), audio.to_vec())?;
        let lengths = Array1::from_vec(vec![audio.len() as i64]);
        let features = self.preprocessor.run(inputs![
            "waveforms" => TensorRef::from_array_view(wav.view())?,
            "waveforms_lens" => TensorRef::from_array_view(lengths.view())?
        ])?;
        let feats = features["features"].try_extract_array::<f32>()?.to_owned();
        let lens = features["features_lens"]
            .try_extract_array::<i64>()?
            .to_owned();
        drop(features);
        let outputs = self.encoder.run(inputs![
            "audio_signal" => TensorRef::from_array_view(feats.view())?,
            "length" => TensorRef::from_array_view(lens.view())?
        ])?;
        let encoded = outputs["encoded"].try_extract_array::<f32>()?.to_owned();
        let steps = outputs["encoded_len"].try_extract_array::<i32>()?[IxDyn(&[0])] as usize;
        drop(outputs);
        if encoded.ndim() != 3 || steps > encoded.shape()[2] {
            bail!("Invalid GigaAM encoder output");
        }
        let mut h = Array3::<f32>::zeros((1, 1, 320)).into_dyn();
        let mut c = h.clone();
        let mut previous = self.blank;
        let mut prediction: Option<ArrayD<f32>> = None;
        let mut next_h = h.clone();
        let mut next_c = c.clone();
        let mut text = String::new();
        for t in 0..steps {
            let frame = encoded.slice(s![0..1, .., t..t + 1]).to_owned();
            for _ in 0..3 {
                if prediction.is_none() {
                    let token = Array2::from_elem((1, 1), previous as i64);
                    let out = self.decoder.run(inputs![
                        "x" => TensorRef::from_array_view(token.view())?,
                        "h.1" => TensorRef::from_array_view(h.view())?,
                        "c.1" => TensorRef::from_array_view(c.view())?
                    ])?;
                    prediction = Some(
                        out["dec"]
                            .try_extract_array::<f32>()?
                            .permuted_axes(IxDyn(&[0, 2, 1]))
                            .to_owned(),
                    );
                    next_h = out["h"].try_extract_array::<f32>()?.to_owned();
                    next_c = out["c"].try_extract_array::<f32>()?.to_owned();
                }
                let out = self.joint.run(inputs![
                    "enc" => TensorRef::from_array_view(frame.view())?,
                    "dec" => TensorRef::from_array_view(prediction.as_ref().unwrap().view())?
                ])?;
                let logits = out["joint"].try_extract_array::<f32>()?;
                let token = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .context("Empty joint output")?
                    .0;
                if token == self.blank {
                    break;
                }
                text.push_str(self.vocab.get(token).context("GigaAM token out of range")?);
                previous = token;
                h = next_h.clone();
                c = next_c.clone();
                prediction = None;
            }
        }
        // Same sentencepiece punctuation/space cleanup as onnx-asr.
        let re = regex::Regex::new(r"\A\s|\s\B|(\s)\b")?;
        Ok(re
            .replace_all(
                &text,
                |caps: &regex::Captures| if caps.get(1).is_some() { " " } else { "" },
            )
            .trim()
            .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Opt-in integration test; model weights remain outside the repository.
    #[test]
    #[ignore = "requires GIGAAM_MODEL_DIR pointing to the verified local model"]
    fn silence_and_invalid_audio() -> Result<()> {
        let dir = std::env::var("GIGAAM_MODEL_DIR")?;
        let mut model = GigaModel::load(Path::new(&dir))?;
        assert_eq!(model.transcribe(&[])?, "");
        assert!(model.transcribe(&[f32::NAN; 1600]).is_err());
        assert_eq!(model.transcribe(&vec![0.0; 16000])?, "");
        Ok(())
    }
}

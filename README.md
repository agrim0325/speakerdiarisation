# Speaker Diarization Pipeline — Python / ML

A Python conversion of the MATLAB G37 SAS speaker turn pipeline.  
The MATLAB signal-processing approach is replaced with a proper ML stack:

| MATLAB (original) | Python ML (this project) |
|---|---|
| Manual FFT + cepstral templates | **librosa** MFCC extraction |
| Hand-rolled K-Means | **scikit-learn** KMeans++ |
| `interp1` resampling | `librosa.load` with native resampling |
| `fft` GCC-PHAT delay | NumPy FFT GCC-PHAT |
| Figure export via MATLAB | **matplotlib** Agg backend |
| `writetable` XLSX | **pandas** + **openpyxl** |

---

## Install

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+

---

## Run

**Default** (uses `Input (Microphone)/` and `Input (Dataset)/`):

```bash
python main.py
```

**Custom paths:**

```bash
python main.py --mic1 path/to/left.mp3 --mic2 path/to/right.mp3
```

**Force speaker count:**

```bash
python main.py --num-speakers 3
```

**All options:**

```
--mic1                  Path to mic 1 file
--mic2                  Path to mic 2 file
--mic-dir               Folder with mic files (default: Input (Microphone))
--dataset-dir           Folder with reference speaker files (default: Input (Dataset))
--output-dir            Output folder (default: Output)
--num-speakers          Force k (auto-detects if omitted)
--min-speakers          Min k for auto-detection (default: 2)
--max-speakers          Max k for auto-detection (default: 5)
--sample-rate           Target sample rate in Hz (default: 16000)
--energy-threshold-pct  VAD energy threshold % of peak (default: 1.5)
--min-segment-duration  Minimum turn duration in seconds (default: 0.20)
--merge-gap             Max gap to merge adjacent turns in seconds (default: 0.50)
```

---

## Project Structure

```
speaker-diarization/
├── main.py                      # Entry point + CLI
├── requirements.txt
├── Input (Microphone)/          # Drop your mic files here
│   ├── a1.mp3
│   └── a2.mp3
├── Input (Dataset)/             # Drop reference speaker recordings here
│   ├── Speaker1.mp3
│   ├── Speaker2.mp3
│   └── Speaker3.mp3
├── Output/                      # Generated automatically
│   ├── combined.wav
│   ├── speaker_timeline.xlsx
│   ├── turn_clips/
│   ├── speaker_clips/
│   ├── reports/
│   │   ├── turns.csv
│   │   └── summary.json
│   └── plots/
│       ├── waveforms_labelled.png
│       └── speaker_clusters.png
└── src/
    ├── pipeline.py              # Orchestrator
    ├── audio.py                 # Loading, mixing, normalization
    ├── segmentation.py          # VAD — short-time energy + Hamming window
    ├── features.py              # MFCC, spectral, pitch, delay features
    ├── models.py                # Diagonal Gaussian speaker models
    ├── clustering.py            # KMeans++ + speaker assignment
    └── output.py                # Clips, CSV, XLSX, JSON, plots
```

---

## ML Concepts Used

- **MFCC feature extraction** — 12 Mel-frequency cepstral coefficients per frame via librosa
- **KMeans++ clustering** — segments grouped into k speaker clusters (scikit-learn)
- **Diagonal Gaussian speaker models** — reference templates from labelled recordings
- **Short-time energy VAD** — Hamming-windowed frame energy for speech detection
- **GCC-PHAT delay estimation** — inter-mic cross-correlation for spatial features
- **Spectral features** — centroid, rolloff, ZCR, flatness per segment
- **pyin pitch estimation** — voiced fundamental frequency per segment
- **Optimal assignment** — exhaustive permutation search to map clusters → speakers

---

## Outputs

| File | Description |
|---|---|
| `combined.wav` | Superposition-mixed mono signal |
| `turn_clips/turn_NN_SpeakerX.wav` | Individual turn audio |
| `speaker_clips/SpeakerX_all_turns.wav` | All turns for each speaker concatenated |
| `speaker_timeline.xlsx` | Turn table with timestamps and scores |
| `reports/turns.csv` | Same as XLSX in CSV format |
| `reports/summary.json` | Full pipeline summary |
| `plots/waveforms_labelled.png` | Waveforms with coloured turn labels |
| `plots/speaker_clusters.png` | Cluster scatter in score space |
# speakerdiarisation

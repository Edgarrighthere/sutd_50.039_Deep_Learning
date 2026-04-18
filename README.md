# PHQ-8 Depression Detection

This repository contains experiments for depression detection on the **DAIC-WOZ / AVEC 2017** benchmark using audio, transcript, and multimodal fusion pipelines.

The main architecture in this project is:

- [`experiments/mid_fusion_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/mid_fusion_mentalRoberta_wavLM.ipynb)

That notebook is the primary reference for the final multimodal model. It performs **mid-fusion of Mental-RoBERTa text embeddings and WavLM audio embeddings**, keeps the unimodal encoders frozen, and trains a fusion MLP on top.

## Project Goal

The task is to predict depression-related labels derived from **PHQ-8** scores using participant speech from DAIC-WOZ interviews.

Across the repo, the experiments cover:

- text-only models built from transcripts
- audio-only models built from participant speech
- classical machine learning baselines
- multimodal fusion models combining text and audio

## Main Model

The main notebook, [`mid_fusion_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/mid_fusion_mentalRoberta_wavLM.ipynb), does the following:

- loads binary PHQ-8 labels for train, dev, and test
- loads pre-extracted text and audio feature caches
- rebuilds the pretrained unimodal encoders
- restores the best Mental-RoBERTa and WavLM checkpoints
- freezes both encoders
- precomputes participant-level embeddings
- trains a **MidFusionClassifier** MLP on the concatenated embeddings
- uses **Optuna** for hyperparameter tuning with validation macro-F1
- tunes the decision threshold on the validation set
- evaluates the final model on the held-out test set

Key idea:

- **Text branch**: Mental-RoBERTa-based encoder
- **Audio branch**: WavLM-based encoder
- **Fusion stage**: concatenate frozen embeddings and train a lightweight classifier

## Dataset

This project uses the **DAIC-WOZ dataset** from the USC Institute for Creative Technologies.

The dataset can be requested here:

- https://dcapswoz.ict.usc.edu/

**To download the dataset upon approval**:
`ensure to use WSL`

```bash
wget -r -np -nH --cut-dirs=1 -R "index.html*" --user=YOUR_USERNAME --ask-password "<link_to_dataset>"
```

In the event of an interrupted download:

```bash
wget -c -r -np -nH --cut-dirs=1 -A "*.zip,*.csv" -R "index.html*" --user=YOUR_USERNAME --ask-password "<link_to_dataset>"
```

The repository does not store the dataset itself. Large raw files, transcripts, and feature caches are ignored in [`.gitignore`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/.gitignore).

## Recommended Workflow

If you are starting from raw data, the notebooks are roughly organized in this order:

1. **Prepare participant audio**
   - [`participant_audio_isolation.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/participant_audio_isolation.ipynb)
   - extracts participant-only speech from DAIC-WOZ session audio using transcript timestamps

2. **Generate transcripts**
   - [`whisper_transcribe.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/whisper_transcribe.ipynb)
   - transcribes `.wav` files with Whisper into `dataset/transcripts/`

3. **Clean transcripts**
   - [`transcript_cleaning_minimal.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/transcript_cleaning_minimal.ipynb)
   - applies lightweight transcript normalization

4. **Process audio**
   - [`audio_process_spectorgram.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/audio_process_spectorgram.ipynb)
   - resamples audio to 16 kHz
   - segments it into fixed windows
   - builds log-mel spectrogram outputs and metadata

5. **Run unimodal or baseline experiments**
   - text models: Mental-RoBERTa, RoBERTa, BERT, DistilBERT
   - audio models: WavLM, wav2vec2, HuBERT, CNN/LSTM variants
   - classical baseline: [`experiments/classical_ml_baselines.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/classical_ml_baselines.ipynb)

6. **Run multimodal fusion**
   - main: [`experiments/mid_fusion_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/mid_fusion_mentalRoberta_wavLM.ipynb)
   - alternatives:
     - [`experiments/early_fusion_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/early_fusion_mentalRoberta_wavLM.ipynb)
     - [`experiments/ensemble_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/ensemble_mentalRoberta_wavLM.ipynb)
     - [`experiments/mid_fusion_temporal_A_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/mid_fusion_temporal_A_mentalRoberta_wavLM.ipynb)

## Important Experiment Notebooks

### Multimodal

- [`experiments/mid_fusion_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/mid_fusion_mentalRoberta_wavLM.ipynb)
  - main architecture
- [`experiments/early_fusion_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/early_fusion_mentalRoberta_wavLM.ipynb)
  - early fusion of raw backbone features before sequence modeling
- [`experiments/ensemble_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/ensemble_mentalRoberta_wavLM.ipynb)
  - late-fusion ensemble of pretrained unimodal models

### Text

- `experiments/mental_roberta_cls.ipynb`
- `experiments/mental_roberta_lstm_cls.ipynb`
- `experiments/roberta_cls.ipynb`
- `experiments/bert_cls.ipynb`
- `experiments/distilbert_cls.ipynb`
- `experiments/distilbert_lstm_cls.ipynb`

### Audio

- `experiments/wavLM_MLP_cls.ipynb`
- `experiments/wavLM_lstm_cls.ipynb`
- `experiments/wav2vec_baseline.ipynb`
- `experiments/wav2vec_classification_aug.ipynb`
- `experiments/wav2vec_finetune.ipynb`
- `experiments/HuBERT_linear.ipynb`
- `experiments/HuBERT_MLP_cls.ipynb`
- `experiments/HuBERT_biLSTM.ipynb`
- `experiments/HuBERT_biLSTM_cls.ipynb`
- `experiments/simple_cnn.ipynb`
- `experiments/cnn_lstm.ipynb`
- `experiments/cnn_biLSTM.ipynb`
- `experiments/cnn_rnn.ipynb`
- `experiments/3_cnn_blocks.ipynb`

### Classical ML

- [`experiments/classical_ml_baselines.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/classical_ml_baselines.ipynb)
  - participant-level handcrafted or aggregated features with conventional ML baselines

## Expected Inputs and Artifacts

The notebooks reference a mixture of raw and derived artifacts, including:

- participant audio files such as `300_P.wav`
- transcript files under `dataset/transcripts/`
- cleaned transcripts
- processed feature caches such as `*.npz`
- saved best checkpoints under `experiments/best_model/`

Some notebooks assume they are run from specific working directories, especially from inside `experiments/`. If paths fail, check the path configuration cells near the top of each notebook.

## Notes

- Several notebooks target **binary classification** from PHQ-8-derived labels.
- Some earlier baselines or exploratory notebooks use **regression** on PHQ-8 scores instead.
- The most complete multimodal training and evaluation pipeline in this repository is the **mid-fusion Mental-RoBERTa + WavLM notebook**.

## Quick Start

If you only want to inspect the main model, start here:

1. Open [`experiments/mid_fusion_mentalRoberta_wavLM.ipynb`](/C:/Users/edgar/OneDrive/Documents/Term_8/Deep_learning/Project/phq8_depression_detection/experiments/mid_fusion_mentalRoberta_wavLM.ipynb)
2. Check the path and checkpoint configuration cells
3. Ensure the required feature caches and pretrained checkpoints exist
4. Run the notebook top to bottom

eir split for simplicity.

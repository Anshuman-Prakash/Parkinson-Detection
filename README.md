# Parkinson Disease Detection Using Speech Signals and Deep Learning

## Overview

This project presents an automated Parkinson's Disease Detection system using speech signals and deep learning techniques. The system analyzes voice recordings and classifies speakers as either **Healthy** or **Parkinson's Affected**.

The primary objective is to develop a **low-cost, non-invasive screening tool** that requires only a microphone, making early Parkinson's screening more accessible.

---

## Problem Statement

Parkinson's Disease affects motor control, movement, and speech production. Traditional diagnosis often requires clinical examination and specialized healthcare facilities.

This project investigates whether Parkinson's Disease can be detected directly from speech recordings using modern deep learning architectures and self-supervised speech models.

---

## Dataset

### PC-GITA Dataset

The project uses the **PC-GITA Speech Corpus**, a widely used benchmark dataset for Parkinson's speech analysis.

Dataset Characteristics:

* Total Samples: **1500**
* Healthy Samples: **750**
* Parkinson's Samples: **750**
* Audio Format: WAV
* Sampling Rate: 16 kHz
* Dataset Split:

  * Training: 80%
  * Testing: 20%

Speech recordings consist primarily of sustained vowel pronunciations.

---

## Data Preprocessing

The following preprocessing steps were applied:

* Stereo to Mono Conversion
* Resampling to 16 kHz
* Amplitude Normalization
* Padding/Truncation to Fixed Length
* Feature Extraction using Self-Supervised Speech Models

Input Shape:

```python
(400, 768)
```

---

## Proposed Architectures

### 1. WavLM + GRU + Attention + SVM

Pipeline:

```text
Raw Audio
   ↓
WavLM Feature Extraction
   ↓
Bidirectional GRU
   ↓
Attention Layer
   ↓
SVM Classifier
   ↓
Prediction
```

### 2. Wav2Vec 2.0 + LSTM + SVM

Pipeline:

```text
Raw Audio
   ↓
Wav2Vec 2.0
   ↓
LSTM
   ↓
SVM
   ↓
Prediction
```

### 3. Wav2Vec 2.0 + GRU + SVM

Pipeline:

```text
Raw Audio
   ↓
Wav2Vec 2.0
   ↓
GRU
   ↓
SVM
   ↓
Prediction
```

---

## Technologies Used

* Python
* PyTorch
* Torchaudio
* Hugging Face Transformers
* Scikit-Learn
* NumPy
* Pandas
* Google Colab

---

## Model Configuration

### LSTM

* Hidden Size: 128
* Layers: 2

### GRU

* Hidden Size: 128
* Layers: 2

### SVM

* Kernel: RBF
* C = 20 (WavLM Pipeline)
* C = 1.0 (LSTM & GRU Pipelines)

---

## Experimental Results

| Model                         | Accuracy   |
| ----------------------------- | ---------- |
| WavLM + GRU + Attention + SVM | **72.33%** |
| Wav2Vec 2.0 + LSTM + SVM      | 66.70%     |
| Wav2Vec 2.0 + GRU + SVM       | 66.33%     |

### Best Performing Model

**WavLM + GRU + Attention + SVM**

Performance Metrics:

* Accuracy: 72.33%
* Precision: 71.79%
* Recall: 74.67%
* F1-Score: 73.20%
* Specificity: 70.00%

---

## Key Findings

* WavLM outperformed Wav2Vec 2.0 for Parkinson's speech classification.
* Attention mechanisms improved diagnostic feature extraction.
* Encoder quality contributed more to performance than choosing between LSTM and GRU.
* Speech-based screening can serve as a practical early detection tool.

---

## Project Structure

```text
Parkinson-Detection/
│
├── GRU.ipynb
├── lstm_svm.ipynb
├── Wavelm+svm.ipynb
└── README.md
```

---

## Future Work

Potential improvements include:

* Fine-tuning WavLM on Parkinson-specific speech
* Multi-class severity prediction
* Real-world noisy environment testing
* Multi-language speech evaluation
* Mobile deployment using ONNX/TensorFlow Lite
* Integration with telemedicine platforms

---

## References

1. PC-GITA Dataset
2. Wav2Vec 2.0 (Meta AI)
3. WavLM: Large-Scale Self-Supervised Pretraining for Speech Processing
4. Scikit-Learn Documentation
5. PyTorch Documentation

---

## Conclusion

This project demonstrates that modern self-supervised speech models can effectively identify Parkinson's-related vocal abnormalities from simple speech recordings. Among all tested architectures, the **WavLM + GRU + Attention + SVM** pipeline achieved the best performance, establishing a strong foundation for future speech-based healthcare applications.

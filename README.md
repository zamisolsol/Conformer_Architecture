# Conformer_Architecture

PyTorch implementation of the **Conformer** model — a hybrid architecture combining CNN and Transformer layers for speech processing tasks like ASR (Automatic Speech Recognition).

---

## 📌 What is this?

This repository contains a modular and clean implementation of the **Conformer** encoder and decoder architecture, including:

- ✅ Macaron-style Feed Forward modules  
- ✅ Multi-head Self-Attention with LayerNorm  
- ✅ Convolution modules with GLU and depthwise separable conv  
- ✅ Subsampling (Conv2D-based 4x reduction)  
- ✅ LSTM-based decoder  
- ✅ Full Conformer Transducer model

---

## 🧱 Model Components

### 🔸 ConformerBlock
Each block is composed of:
1. Half-step FFN
2. Multi-head Self-Attention (MHSA)
3. Convolution Module
4. Another Half-step FFN
5. Final LayerNorm

### 🔸 ConformerEncoder
- Subsampling audio features using 2-layer Conv2D
- Stack of multiple `ConformerBlock` modules

### 🔸 LSTMDecoder
- Simple LSTM-based decoder layer
- Projects encoder outputs into vocabulary space

---

## 📂 Code Structure

| File | Description |
|------|-------------|
| `ConformerTransducer` | Full model combining encoder & decoder |
| `ConformerEncoder`    | Stack of Conformer blocks |
| `ConformerBlock`      | Core building block |
| `ConvSubsampling`     | 4x temporal reduction using Conv2D |
| `LSTMDecoder`         | Projects output to vocab space |

---

## 🔧 How to Use

```python
import torch
from conformer import ConformerTransducer

model = ConformerTransducer(input_dim=80, vocab_size=1000)
x = torch.randn(2, 200, 80)  # batch_size x time_steps x feature_dim
out = model(x)  # shape: [2, T', vocab_size]

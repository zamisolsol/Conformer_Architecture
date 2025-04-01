# Conformer_Architecture

PyTorch로 구현한 **Conformer 모델**입니다.  
Conformer는 CNN과 Transformer의 장점을 결합한 하이브리드 구조로,  
음성 인식(ASR, Automatic Speech Recognition)과 같은 음성 처리 작업에 적합한 모델입니다.

---

## 📌 이 저장소는?

이 저장소에는 Conformer 인코더 및 디코더 아키텍처를 깔끔하고 모듈화된 방식으로 구현한 코드가 포함되어 있습니다.

- ✅ Macaron 구조의 Feed Forward 모듈
- ✅ LayerNorm을 포함한 Multi-head Self-Attention
- ✅ GLU + Depthwise Convolution이 포함된 컨볼루션 모듈
- ✅ 4배 시간 축 축소(Conv2D 기반 Subsampling)
- ✅ 간단한 LSTM 기반 디코더
- ✅ 인코더와 디코더가 결합된 전체 Conformer Transducer 모델

---

## 🧱 모델 구성

### 🔸 ConformerBlock
각 블록은 다음과 같이 구성됩니다:
1. 절반 FFN (Macaron 구조)
2. 다중 헤드 자기 주의 (MHSA)
3. 컨볼루션 모듈
4. 또 다른 절반 FFN
5. 최종 LayerNorm

### 🔸 ConformerEncoder
- 2개의 Conv2D 층으로 입력을 서브샘플링 (4배 축소)
- 여러 개의 `ConformerBlock`을 순차적으로 연결

### 🔸 LSTMDecoder
- 간단한 LSTM 계층으로 구성된 디코더
- 인코더 출력을 어휘 공간으로 투사

---

## 📂 코드 구조

| 파일 | 설명 |
|------|------|
| `ConformerTransducer` | 인코더 + 디코더를 결합한 전체 모델 |
| `ConformerEncoder` | 여러 Conformer 블록으로 구성된 인코더 |
| `ConformerBlock` | 핵심 연산 블록 |
| `ConvSubsampling` | Conv2D 기반 시간 축 축소 |
| `LSTMDecoder` | 어휘 예측을 위한 디코더 |

---

## 🔧 사용 예시

```python
import torch
from conformer import ConformerTransducer

model = ConformerTransducer(input_dim=80, vocab_size=1000)
x = torch.randn(2, 200, 80)  # 배치 x 시간 x 입력 피처
out = model(x)  # 결과: [2, T', vocab_size] 형태

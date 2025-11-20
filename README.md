# 🧵 온라인 리뷰 특화 한국어 자연어 처리 모델 구현: ReBERT, ReELECTRA
### **ReBERT / ReELECTRA: Domain-Adaptive Fine-tuning for Online Reviews**

---

## 1. 📘 프로젝트 개요

최근 전자상거래 시장이 급속히 성장함에 따라 **온라인 고객 리뷰(OCR, Online Customer Review)**는 소비자의 구매 결정에 큰 영향을 끼치는 핵심 요인이 되었습니다.  
본 프로젝트는 **온라인 패션 플랫폼 리뷰 데이터**를 기반으로 감성 분석 모델을 구축하고, 이를 통해 리뷰에 담긴 고객의 긍‧부정 감성을 자동으로 분류하는 것을 목표로 합니다.

이를 위해 범용 사전학습 언어모델인 **BERT**와 **ELECTRA**를 패션 리뷰 도메인에 최적화되도록 다시 학습시켜 각각:

- **ReBERT (Review-BERT)**  
- **ReELECTRA (Review-ELECTRA)**  

두 모델을 구현하였습니다.

---

## 2. 🛠️ 사용 기술 스택

### ✔ Modeling
- PyTorch  
- Transformers (HuggingFace)  
- Tokenizers  

### ✔ Preprocessing
- soynlp  
- emoji  

### ✔ Crawling
- selenium  

### ✔ Data Analysis
- pandas  
- numpy  
- tqdm  

> 전체 패키지 목록은 `requirements.txt` 참고.

---

## 3. 📁 폴더 구조

📂 project/

<details>
<summary>📂 data/</summary>

```
├── raw/                                 # 원본
│   ├── model/                           # 모델 학습용
│   │   ├── 📄 pretraining.txt
│   │   ├── 📄 dapt.txt
│   │   └── 📄 finetuning.txt
│   │
│   └── review/                          # 리뷰 데이터
│       └── 📄 musinsa_review_{goods_no}.csv
│
└── processed/                           # 텍스트 전처리
    ├── model/                           # 모델 학습용
    │   ├── 📄 pretraining_preprocessed.txt
    │   ├── 📄 dapt_preprocessed.txt
    │   └── 📄 finetuning_preprocessed.txt
    │
    └── review/                          # 감성 분류
        ├── ELECTRA/
        │   └── 📄 labeled_review_{goods_no}.csv
        └── BERT/
            └── 📄 labeled_review_{goods_no}.csv
```

</details>

<details>
<summary>📂 model/</summary>

```
├── ReBERT/
│   ├── checkpoints/                # 체크포인트
│   ├── pretrained/                 # 사전학습 모델
│   ├── DAPT/                       # DAPT 모델
│   └── finetuned/                  # 파인튜닝 모델
│
├── ReELECTRA/
│   ├── checkpoints/                # 체크포인트
│   ├── pretrained/                 # 사전학습 모델
│   ├── DAPT/                       # DAPT 모델
│   └── finetuned/                  # 파인튜닝 모델
│
├── KcBERT/
│   ├── checkpoints/                # 체크포인트
│   └── finetuned/                  # 파인튜닝 모델
│
└── KcELECTRA/
    ├── checkpoints/                # 체크포인트
    └── finetuned/                  # 파인튜닝 모델
```

</details>

<details>
<summary>📂 src/</summary>

```
├── classification.py
├── crawling.py
├── KcBERT.py
├── KcELECTRA.py
├── preprocessing.py
├── tokenizer.py
│
├── ReBERT/
│   ├── pretraining.py
│   ├── DAPT.py
│   └── finetuning.py
│
└── ReELECTRA/
    ├── pretraining.py
    ├── DAPT.py
    └── finetuning.py
```

</details>

<details>
<summary>📄 requirements.txt</summary>
</details>

<details>
<summary>📄 README.md</summary>
</details>

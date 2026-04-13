# QLoRA-for-Finance Portfolio Project Roadmap

This is a **strong portfolio topic**.  
A finance-domain QLoRA project lets you showcase:

- **domain adaptation**
- **efficient fine-tuning**
- **quantization tradeoffs**
- **evaluation beyond generic NLP metrics**
- **deployment / MLOps thinking**

Below I’ll give you a **practical, research-grade roadmap** with concrete datasets, models, repos, and implementation patterns.

---

# 1. Datasets and benchmark tasks

## A. Best task framing for a portfolio project

If your goal is a **convincing end-to-end profile project**, I recommend this main task:

### **Primary portfolio task**
**“Earnings-call / SEC risk summarization + risk-level classification”**

Why this is good:

- financially realistic
- long-document friendly
- easy to justify domain adaptation
- naturally supports:
  - summarization
  - classification
  - factuality / entity-preservation evaluation
  - quantization tradeoff analysis

### Good secondary tasks
1. **Financial news sentiment / market tone classification**  
2. **SEC filing QA / long-context reasoning**
3. **Trader/client or analyst-style dialogue summarization**
4. **Compliance / regulatory answer generation**
5. **Instruction tuning on financial QA / advice-style prompts**

---

## B. Recommended datasets by task

I’m prioritizing datasets that are available on Hugging Face, and especially those useful for **instruction tuning**, **classification**, **summarization**, or **financial reasoning**.

---

## 1) Earnings-report summarization / analysis

### **Bose345/sp500_earnings_transcripts**
A strong candidate for your main corpus. The dataset card says it contains S&P 500 and U.S. large-cap earnings call transcripts spanning **2005–2025**, with transcript-scale text suitable for summarization, sentiment/risk analysis, and long-context experiments. ([huggingface.co](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts?utm_source=openai))

**HF dataset ID**
- `Bose345/sp500_earnings_transcripts`

**Why it’s good**
- large-ish, domain-pure corpus
- realistic company language
- ideal for:
  - summarization
  - risk cue extraction
  - event / outlook classification
  - long-context chunking experiments
- very good for creating your own **instruction pairs**

**How to use it**
- build:
  - transcript → executive summary
  - transcript section → risk-level label
  - transcript → “top 5 risks / catalysts”

---

### **finosfoundation/EarningsCallTranscript**
This one is more **audio/transcription-oriented** than classic summarization, but still useful if you want to show a pipeline from audio to transcript to summary. The dataset card describes segmented earnings-call audio paired with transcriptions. ([huggingface.co](https://huggingface.co/datasets/finosfoundation/EarningsCallTranscript?utm_source=openai))

**HF dataset ID**
- `finosfoundation/EarningsCallTranscript`

**Best use**
- ASR-to-summary pipeline
- speech/text benchmarking
- less ideal than `Bose345/sp500_earnings_transcripts` for pure text fine-tuning

---

### **JanosAudran/financial-reports-sec**
SEC-style financial reports on Hugging Face; useful as a source corpus for summarization or instruction generation. ([huggingface.co](https://huggingface.co/datasets/JanosAudran/financial-reports-sec?utm_source=openai))

**HF dataset ID**
- `JanosAudran/financial-reports-sec`

**Best use**
- 10-K / 10-Q section summarization
- risk factor extraction
- MD&A summarization
- compliance-style prompting

---

### **nvidia/Nemotron-SpecializedDomains-Finance-v1**
This dataset is especially interesting because the card says it draws from **SEC EDGAR filings (10-K and 10-Q) from S&P 500 companies, 2019–2024** and is part of NVIDIA’s specialized finance-domain collection. ([huggingface.co](https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1?utm_source=openai))

**HF dataset ID**
- `nvidia/Nemotron-SpecializedDomains-Finance-v1`

**Best use**
- domain adaptation
- SEC-based instruction tuning
- retrieval/QA style experiments
- professional analyst-style financial reasoning

---

## 2) Risk-level classification

There is no single universally standard “high / medium / low risk” HF benchmark in finance, so for your project the most realistic approach is:

### **Create a risk-level benchmark from SEC / earnings data**
Use:

- `Bose345/sp500_earnings_transcripts`
- `JanosAudran/financial-reports-sec`
- `nvidia/Nemotron-SpecializedDomains-Finance-v1`

and derive labels from:
- Risk Factors sections
- management outlook language
- legal / liquidity / leverage indicators
- weak guidance / going-concern signals
- analyst Q&A stress language

This is actually a **better portfolio move** than relying on a tiny canned dataset, because it shows dataset design skill.

### Labeling strategy
Define:

- **High risk**:
  - bankruptcy / going concern
  - severe litigation/regulatory issues
  - major revenue collapse
  - covenant/liquidity issues
  - strong negative forward guidance

- **Medium risk**:
  - moderate uncertainty
  - margin pressure
  - sector weakness
  - mixed guidance
  - execution risk

- **Low risk**:
  - stable guidance
  - strong balance sheet
  - low uncertainty language
  - resilient demand / manageable exposure

You can bootstrap labels using:
1. rule-based weak supervision
2. LLM-assisted annotation
3. human spot-audit

Then train:
- classifier head model
- or instruction model returning `{risk_level, rationale, evidence_span}`

---

## 3) Financial news sentiment

### **lmassaron/FinancialPhraseBank**
This is one of the most standard finance sentiment datasets on Hugging Face. The dataset card states it contains about **4,840–4,846** financial news sentences/headlines with **negative / neutral / positive** labels. ([huggingface.co](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank?utm_source=openai))

**HF dataset ID**
- `lmassaron/FinancialPhraseBank`

Also mirrored variants exist, such as `FinanceMTEB/financial_phrasebank` and `chinua31/financial_phrasebank`. ([huggingface.co](https://huggingface.co/datasets/FinanceMTEB/financial_phrasebank?utm_source=openai))

**Why use it**
- standard baseline
- easy classification benchmark
- great for sanity-check experiments before long-document work

---

### **LLukas22/fiqa**
A Hugging Face-preprocessed version of FiQA is available. ([huggingface.co](https://huggingface.co/datasets/LLukas22/fiqa?utm_source=openai))

**HF dataset ID**
- `LLukas22/fiqa`

**Why use it**
- financial sentiment / QA related benchmark family
- useful for market-oriented text understanding
- good supplementary benchmark to show transfer

---

### **NOSIBLE/financial-sentiment**
Another HF finance-sentiment dataset, and the card claims models trained on it can outperform models trained only on Financial PhraseBank in some setups. ([huggingface.co](https://huggingface.co/datasets/NOSIBLE/financial-sentiment?utm_source=openai))

**HF dataset ID**
- `NOSIBLE/financial-sentiment`

**Use**
- additional classification training data
- cross-dataset generalization evaluation

---

## 4) Instruction tuning / Alpaca-style finance datasets

### **hongfenglu/finance-alpaca**
This is directly in the style you asked for: an Alpaca-style financial instruction dataset on Hugging Face. The dataset card shows around **64.9k** training rows with `prompt` and `completion`. ([huggingface.co](https://huggingface.co/datasets/hongfenglu/finance-alpaca?utm_source=openai))

**HF dataset ID**
- `hongfenglu/finance-alpaca`

**Why it’s useful**
- ideal for SFT / QLoRA
- good if you want a chatbot-like finance assistant
- useful as a warm-start before fine-tuning on your specific task

---

### **sujet-ai/Sujet-Finance-Instruct-177k**
This is one of the strongest instruction datasets for your use case. The dataset card says it contains **177,597** de-duplicated entries aggregated from **18 Hugging Face datasets** across finance tasks such as sentiment analysis, QA, contextual QA, and QA conversation. It explicitly includes `gbharti/finance-alpaca` as a source. ([huggingface.co](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k?utm_source=openai))

**HF dataset ID**
- `sujet-ai/Sujet-Finance-Instruct-177k`

**Why it may be your best finance instruction dataset**
- large scale
- mixed-task
- domain-specialized
- very suitable for QLoRA SFT
- better than a tiny single-purpose corpus if you want a portfolio-ready instruct model

---

### **Josephgflowers/Finance-Instruct-500k**
This appears to be a larger finance-instruction collection/viewer on Hugging Face, with finance-related filtering and Alpaca-like conversion patterns shown in the card. ([huggingface.co](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k/viewer/default/train?utm_source=openai))

**HF dataset ID**
- `Josephgflowers/Finance-Instruct-500k`

**Caution**
- inspect carefully for noise / duplication / quality consistency before training

---

## 5) SEC / long-context reasoning benchmarks

### **GGLabYale/Fin-RATE**
Very strong for evaluation. The dataset card describes it as a **7,500-instance benchmark** for professional-grade reasoning over SEC filings, including long-context understanding, cross-year tracking, and cross-company comparison. ([huggingface.co](https://huggingface.co/datasets/GGLabYale/Fin-RATE?utm_source=openai))

**HF dataset ID**
- `GGLabYale/Fin-RATE`

**Why it matters**
- excellent benchmark for “financial reasoning” claims
- good eval set even if your fine-tuning data is different

---

### **Financbase/financbase-10k-jsonl**
A finance/SEC/investment-analysis-oriented resource hosted on HF. ([huggingface.co](https://huggingface.co/Financbase/financbase-10k-jsonl?utm_source=openai))

**HF ID**
- `Financbase/financbase-10k-jsonl`

**Best use**
- retrieval + QA
- filing-based question answering
- long-context RAG baseline comparisons

---

## 6) Compliance-style / regulatory data

There are fewer clean standard HF datasets here than for sentiment. For a portfolio project, a practical route is:

- use SEC/regulatory corpora:
  - `JanosAudran/financial-reports-sec`
  - `nvidia/Nemotron-SpecializedDomains-Finance-v1`
- optionally combine with a regulatory QA-style dataset such as the Hugging Face regulatory QA example dataset page surfaced in search results. ([huggingface.co](https://huggingface.co/datasets/Magneto/qa_with_personas?utm_source=openai))

But I would frame this as:
### **“compliance-style answer generation over SEC / disclosure text”**
rather than claiming strict legal compliance benchmarking.

---

## 7) Trader-client dialogue summarization

This is the weakest area for standard public HF finance benchmarks. Public finance dialogue corpora are limited.

So I recommend:
- use **instruction-style finance QA datasets** for dialogue fine-tuning:
  - `sujet-ai/Sujet-Finance-Instruct-177k`
  - `hongfenglu/finance-alpaca`
- create a synthetic but realistic dataset:
  - analyst ↔ client
  - RM ↔ investor
  - compliance ↔ trader

That is acceptable for a portfolio if you clearly state:
- synthetic dialogue generation method
- review / filtering
- limitations

---

## C. My shortlist: best datasets for your project

If you want a focused stack, I’d use:

### **Core training corpus**
1. `sujet-ai/Sujet-Finance-Instruct-177k` — best broad financial instruction dataset ([huggingface.co](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k?utm_source=openai))
2. `Bose345/sp500_earnings_transcripts` — best main raw corpus for earnings/risk summarization project ([huggingface.co](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts?utm_source=openai))
3. `nvidia/Nemotron-SpecializedDomains-Finance-v1` — strong SEC/domain-specialized corpus ([huggingface.co](https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1?utm_source=openai))

### **Benchmark / eval**
4. `lmassaron/FinancialPhraseBank` — standard sentiment baseline ([huggingface.co](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank?utm_source=openai))
5. `LLukas22/fiqa` — supplementary finance eval ([huggingface.co](https://huggingface.co/datasets/LLukas22/fiqa?utm_source=openai))
6. `GGLabYale/Fin-RATE` — long-context SEC reasoning benchmark ([huggingface.co](https://huggingface.co/datasets/GGLabYale/Fin-RATE?utm_source=openai))

---

## D. “Best large-scale financial dataset” recommendation

If you want **one answer**:

### Best large-scale instruction dataset
**`sujet-ai/Sujet-Finance-Instruct-177k`**  
because it is large, finance-specific, multi-task, and already instruction-structured. ([huggingface.co](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k?utm_source=openai))

### Best large-scale raw corpus for your specific portfolio project
**`Bose345/sp500_earnings_transcripts`**  
because it aligns directly with earnings analysis, summarization, and risk classification. ([huggingface.co](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts?utm_source=openai))

### Best SEC-focused reasoning/evaluation dataset
**`GGLabYale/Fin-RATE`** and **`nvidia/Nemotron-SpecializedDomains-Finance-v1`** for SEC-centric workflows. ([huggingface.co](https://huggingface.co/datasets/GGLabYale/Fin-RATE?utm_source=openai))

---

# 2. Model recommendations

You asked for “latest open-source LLMs (2015–2026)” and models in the **1B–10B or less** range. For current practical work, the relevant models are modern 2024–2026 open-weight models; anything from 2015 is historically interesting but not competitive for this project.

## Best practical models for your project

### 1. **Qwen/Qwen2.5-3B-Instruct**
A very strong small model. The model card states it has **3.09B parameters**, long-context support, and strong structured-output behavior. ([huggingface.co](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct?utm_source=openai))

**HF ID**
- `Qwen/Qwen2.5-3B-Instruct`

**Why I like it**
- excellent size/performance ratio
- strong instruction-following
- good for JSON outputs
- good candidate for 4-bit QLoRA
- realistic single-GPU fine-tuning

---

### 2. **microsoft/Phi-3.5-mini-instruct**
The card describes it as a lightweight model with **128K context** and notes competitiveness with larger open models. ([huggingface.co](https://huggingface.co/microsoft/Phi-3.5-mini-instruct?utm_source=openai))

**HF ID**
- `microsoft/Phi-3.5-mini-instruct`

**Why use it**
- very strong small-model baseline
- good for edge-ish deployment
- widely used in compact setups

For on-device deployment, Qualcomm also provides an optimized variant and explicitly notes quantized deployment details such as **w4a16 / w8a16**. ([huggingface.co](https://huggingface.co/qualcomm/Phi-3.5-mini-instruct?utm_source=openai))

---

### 3. **google/gemma-2-2b-it**
A lightweight instruct model from Google’s Gemma family. ([huggingface.co](https://huggingface.co/google/gemma-2-2b-it?utm_source=openai))

**HF ID**
- `google/gemma-2-2b-it`

**Why use it**
- compact
- nice for CPU / low-VRAM experiments
- good to compare against Qwen/Phi on finance adaptation

---

### 4. **meta-llama/Llama-3.2-1B-Instruct**
A very small instruct model, useful if you want an extreme edge baseline. The model card points users to quantized/local recipes. ([huggingface.co](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct?utm_source=openai))

**HF ID**
- `meta-llama/Llama-3.2-1B-Instruct`

**Use**
- low-resource baseline
- CPU-ish testing
- “how far can tiny models go in finance?” experiment

---

### 5. **meta-llama/Llama-3.2-3B-Instruct**
A stronger compact Llama option. HF references for this family are available. ([huggingface.co](https://huggingface.co/mendeza/Llama-3.2-3B-Instruct?utm_source=openai))

**HF ID**
- `meta-llama/Llama-3.2-3B-Instruct`

**Use**
- good mainstream baseline
- solid ecosystem support
- strong comparison target

---

### 6. **ibm-granite/granite-3.1-2b-instruct**
This model card describes a long-context **2B-class instruct model** for enterprise/business applications, including summarization, extraction, QA, and long-document tasks. ([huggingface.co](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct?utm_source=openai))

**HF ID**
- `ibm-granite/granite-3.1-2b-instruct`

**Why it’s interesting**
- enterprise positioning
- long context
- compact
- good fit for finance/business docs

A newer version exists according to the card: `ibm-granite/granite-3.3-2b-instruct`. ([huggingface.co](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct?utm_source=openai))

---

### 7. **mistralai/Mistral-7B-Instruct-v0.3**
Still a very practical 7B-class choice and heavily used in QLoRA workflows. Quantized derivatives are widespread on HF. ([huggingface.co](https://huggingface.co/rapidfire-ai-inc/Mistral-7B-Instruct-v0.3-bnb-4bit?utm_source=openai))

**HF family**
- `mistralai/Mistral-7B-Instruct-v0.3`

**Why use it**
- stable ecosystem
- strong 7B baseline
- widely supported in PEFT/TRL examples

---

### 8. **Qwen/Qwen2.5-1.5B-Instruct** or nearby smaller Qwen variants
The Qwen2.5 family includes multiple sizes from **0.5B to 72B** according to the model card. ([huggingface.co](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct?utm_source=openai))

**Why useful**
- lets you build a scaling study:
  - 1.5B vs 3B vs 7B
- ideal for a quantization/efficiency project

---

## Recommended final shortlist

### Best overall for your portfolio
1. **Qwen/Qwen2.5-3B-Instruct**
2. **microsoft/Phi-3.5-mini-instruct**
3. **meta-llama/Llama-3.2-3B-Instruct**
4. **google/gemma-2-2b-it**
5. **mistralai/Mistral-7B-Instruct-v0.3**
6. **ibm-granite/granite-3.1-2b-instruct**

---

## Which one should you actually pick?

### If you want the safest best choice:
**Qwen2.5-3B-Instruct**

### If you want strongest edge/on-device flavor:
**Phi-3.5-mini-instruct** or **Granite 2B**

### If you want strongest community familiarity:
**Mistral-7B-Instruct-v0.3** or **Llama-3.2-3B-Instruct**

---

# 3. Quantization / QLoRA compatibility

The Hugging Face PEFT quantization docs explicitly describe QLoRA-style fine-tuning with:

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_use_double_quant=True`
- `bnb_4bit_compute_dtype=torch.bfloat16`

and recommend `prepare_model_for_kbit_training()` before attaching LoRA adapters. The docs also note that `target_modules="all-linear"` is generally best for QLoRA-style training. ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

Hugging Face TRL docs also describe combining 4-bit loading with PEFT for QLoRA. ([huggingface.co](https://huggingface.co/docs/trl/v1.0.0/en/peft_integration?utm_source=openai))

Transformers docs cover bitsandbytes-based 8-bit and 4-bit quantization, and also point to GPTQ-style options. ([huggingface.co](https://huggingface.co/docs/transformers/v4.51.1/quantization/bitsandbytes?utm_source=openai))

So for your project, the most defensible quantization stack is:

- **training**: bitsandbytes 4-bit NF4 + PEFT LoRA
- **inference comparisons**:
  - FP16 / BF16
  - 8-bit bitsandbytes
  - 4-bit NF4
  - optionally GPTQ post-training inference

---

# 4. Papers / docs you should cite in the project

## Core papers / docs

### **QLoRA paper**
The official GitHub repo summarizes the method and links the paper:  
**“QLoRA: Efficient Finetuning of Quantized LLMs”**. It introduces NF4, double quantization, and paged optimizers. ([github.com](https://github.com/artidoro/qlora?utm_source=openai))

### **LoRA paper**
LoRA is the foundational low-rank adaptation method; use the original paper as your conceptual baseline. ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

### **Hugging Face PEFT quantization docs**
Best practical implementation reference. ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

### **Transformers bitsandbytes quantization docs**
Best practical inference/loading reference. ([huggingface.co](https://huggingface.co/docs/transformers/v4.51.1/quantization/bitsandbytes?utm_source=openai))

### **Fin-RATE benchmark**
Use for financial long-context evaluation framing. ([huggingface.co](https://huggingface.co/datasets/GGLabYale/Fin-RATE?utm_source=openai))

---

# 5. Public notebooks / repos you can use

## QLoRA / LoRA repos

### **artidoro/qlora**
The canonical repository for QLoRA. Strong must-cite repo for your project. ([github.com](https://github.com/artidoro/qlora?utm_source=openai))

### **Hugging Face PEFT docs**
Implementation recipes for quantized PEFT training. ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

### **Hugging Face TRL PEFT integration**
For SFT workflows and script templates. ([huggingface.co](https://huggingface.co/docs/trl/v1.0.0/en/peft_integration?utm_source=openai))

---

## What to search/use in practice
For reproducible portfolio code, I’d base your implementation on:

- `transformers`
- `peft`
- `trl`
- `accelerate`
- `bitsandbytes`

with the official docs/repo above as the foundation. ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

---

# 6. End-to-end project plan

---

## Phase 1 – Problem definition and data setup

## Recommended project title
### **“QLoRA Optimization for Financial Risk Summarization and Classification on Earnings Calls and SEC Filings”**

## Define the main task
I recommend a **multi-task setup**:

### Task A: Summarization
Input:
- earnings call transcript chunk
- SEC filing section

Output:
- concise finance summary
- key risks
- outlook
- material entities

### Task B: Risk-level classification
Input:
- transcript section / risk-factor section

Output:
- `low`, `medium`, `high`

Optional:
- rationale
- evidence span

---

## Dataset setup

### Training datasets
- `sujet-ai/Sujet-Finance-Instruct-177k` for general finance instruction tuning ([huggingface.co](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k?utm_source=openai))
- `Bose345/sp500_earnings_transcripts` for domain corpus and custom task creation ([huggingface.co](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts?utm_source=openai))
- `nvidia/Nemotron-SpecializedDomains-Finance-v1` or `JanosAudran/financial-reports-sec` for SEC text ([huggingface.co](https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1?utm_source=openai))

### Evaluation datasets
- `lmassaron/FinancialPhraseBank` for sentiment baseline/generalization ([huggingface.co](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank?utm_source=openai))
- `GGLabYale/Fin-RATE` for filing reasoning benchmark ([huggingface.co](https://huggingface.co/datasets/GGLabYale/Fin-RATE?utm_source=openai))

---

## Data preprocessing workflow

### Step 1: Normalize fields
Convert all datasets to a common schema:

```python
{
  "source": "...",
  "task": "summarization|classification|qa|instruction",
  "input_text": "...",
  "target_text": "...",
  "label": "...",   # optional
  "metadata": {...}
}
```

### Step 2: Chunk long documents
For earnings calls / SEC filings:
- chunk by section headers first
- then token-length chunking
- retain overlap: 128–256 tokens
- store:
  - `company`
  - `quarter`
  - `year`
  - `section`
  - `ticker`

### Step 3: Build labels for risk classification
Options:
1. heuristic weak labels
2. LLM-assisted initial labels
3. manual audit for 500–2,000 examples

### Step 4: Train/dev/test split
Use **time-aware splits** if possible:
- train: older years
- dev: later years
- test: most recent years

This is better than random split for finance because domain drift is real.

### Example split
- train: 2015–2022
- dev: 2023
- test: 2024–2025

For small benchmark datasets like FinancialPhraseBank:
- stratified random split

---

## Example preprocessing pseudocode

```python
from datasets import load_dataset, DatasetDict
import re

ds = load_dataset("Bose345/sp500_earnings_transcripts")

def clean_text(x):
    text = x["text"]
    text = re.sub(r"\s+", " ", text).strip()
    return {"text": text}

ds = ds.map(clean_text)

def make_summary_example(row):
    prompt = (
        "You are a financial analyst.\n"
        "Summarize the earnings call transcript. "
        "Extract key risks, guidance, and management outlook."
    )
    return {
        "instruction": prompt,
        "input": row["text"][:12000],  # or chunked input
        "output": row.get("summary", "")
    }

train_ds = ds["train"].map(make_summary_example)
```

---

## Phase 2 – QLoRA / LoRA fine-tuning design

## Base model choice

### Best practical default:
**`Qwen/Qwen2.5-3B-Instruct`** ([huggingface.co](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct?utm_source=openai))

### Alternative matrix
- **tiny baseline**: `meta-llama/Llama-3.2-1B-Instruct` ([huggingface.co](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct?utm_source=openai))
- **edge baseline**: `google/gemma-2-2b-it` or `ibm-granite/granite-3.1-2b-instruct` ([huggingface.co](https://huggingface.co/google/gemma-2-2b-it?utm_source=openai))
- **stronger baseline**: `mistralai/Mistral-7B-Instruct-v0.3` ([huggingface.co](https://huggingface.co/rapidfire-ai-inc/Mistral-7B-Instruct-v0.3-bnb-4bit?utm_source=openai))

---

## QLoRA config recommendation

Based on HF PEFT docs, use 4-bit NF4 plus double quantization. ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

### Suggested config
- quantization:
  - `load_in_4bit=True`
  - `bnb_4bit_quant_type="nf4"`
  - `bnb_4bit_use_double_quant=True`
  - `bnb_4bit_compute_dtype=torch.bfloat16`
- LoRA:
  - `r=16` or `32`
  - `lora_alpha=32` or `64`
  - `lora_dropout=0.05`
  - `target_modules="all-linear"`
- optimizer:
  - paged AdamW if available / standard AdamW
- max sequence:
  - 2k–4k for training
- grad accumulation:
  - set to fit single GPU

---

## Prompt template design

### Summarization template

```text
### Instruction:
You are a financial analyst. Summarize the following earnings-call or SEC text.

Return:
1. Executive summary
2. Key risks
3. Forward outlook
4. Important entities and figures
5. Risk level: low / medium / high

### Input:
{document_chunk}

### Response:
```

### Classification template

```text
### Instruction:
Classify the financial risk level of the following disclosure text.

Labels:
- low
- medium
- high

Also provide a one-sentence rationale citing textual evidence.

### Input:
{text}

### Response:
Risk level:
Rationale:
```

### JSON-safe template
Very useful if you want robust evaluation.

```text
Return valid JSON with keys:
summary, key_risks, outlook, entities, risk_level
```

Qwen models are especially good with structured outputs according to the model card. ([huggingface.co](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct?utm_source=openai))

---

## Example QLoRA training script

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer

model_id = "Qwen/Qwen2.5-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")

def format_example(ex):
    prompt = f"""### Instruction:
{ex.get('instruction', ex.get('prompt', 'Answer the finance question.'))}

### Input:
{ex.get('input', '')}

### Response:
{ex.get('output', ex.get('completion', ''))}"""
    return {"text": prompt}

dataset = dataset.map(format_example)

args = TrainingArguments(
    output_dir="./qwen-finance-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=20,
    save_steps=500,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    peft_config=peft_config,
    formatting_func=lambda x: x["text"],
)

trainer.train()
model.save_pretrained("./qwen-finance-qlora-adapter")
tokenizer.save_pretrained("./qwen-finance-qlora-adapter")
```

This setup is consistent with HF PEFT/TRL guidance for quantized LoRA/QLoRA workflows. ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

---

## Phase 3 – Quantization strategies

You want this phase to be a real experiment, not just implementation.

## Compare:
1. **FP16/BF16 full model inference**
2. **8-bit bitsandbytes**
3. **4-bit NF4 QLoRA-trained**
4. **optional GPTQ post-training quantized inference**

HF docs describe bitsandbytes 8-bit/4-bit loading and GPTQ support in the quantization guides. ([huggingface.co](https://huggingface.co/docs/transformers/v4.51.1/quantization/bitsandbytes?utm_source=openai))

---

## Evaluation matrix

| Variant | Trainable? | VRAM | Latency | Quality |
|---|---:|---:|---:|---:|
| FP16 full | yes/no | highest | medium | strongest baseline |
| 8-bit | inference / some PEFT | medium | good | near-baseline |
| 4-bit NF4 + LoRA | yes | low | good/medium | often best efficiency tradeoff |
| GPTQ 4-bit | mostly inference-focused | low | often strong | may vary |

---

## Example loading code

### 8-bit inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-3B-Instruct"

bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_8bit,
    device_map="auto"
)
```

### 4-bit NF4 inference

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_4bit,
    device_map="auto"
)
```

This directly matches the PEFT/Transformers quantization guidance. ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

---

## What to report
For each variant, record:

- max GPU memory
- tokens/sec
- average latency
- exact-match / classification F1
- ROUGE / BERTScore
- factual consistency / entity preservation
- failure modes:
  - dropped numbers
  - wrong company names
  - inverted risk labels

---

## Phase 4 – Evaluation and fintech-style metrics

This is where your portfolio can stand out.

## Standard metrics

### Summarization
- ROUGE-1 / ROUGE-L
- BERTScore
- compression ratio

### Classification
- accuracy
- macro F1
- confusion matrix
- class-wise precision/recall

---

## Finance-specific metrics

### 1. Risk-level classification F1
Primary for the classification task.

### 2. Entity preservation
Measure whether the summary preserves:
- company
- ticker
- dates/quarters
- revenue / EPS / guidance values

Can be done via:
- regex
- NER matching
- structured extraction comparison

### 3. Numerical faithfulness
For finance, numbers matter more than prose style.

Track:
- exact numeric match rate
- signed-direction correctness:
  - growth vs decline
  - beat vs miss
- hallucinated number count

### 4. Fact consistency
Use:
- LLM-as-judge with strict rubric
- or NLI-style entailment between source chunk and generated summary

### 5. Evidence grounding
If your classifier produces rationale:
- measure whether cited spans actually support the label

---

## Example evaluation schema

```python
{
  "model_version": "qwen2.5-3b-finance-qlora-v1",
  "quantization": "4bit_nf4",
  "task": "risk_summary",
  "metrics": {
    "macro_f1": 0.81,
    "rougeL": 0.37,
    "entity_preservation": 0.92,
    "numeric_faithfulness": 0.88,
    "avg_latency_ms": 740,
    "max_vram_gb": 6.4
  }
}
```

---

## MLOps-style evaluation tracking

Use one:
- **MLflow**
- **Weights & Biases**
- **ClearML**

Track:
- dataset version
- preprocessing version
- base model
- adapter config
- quantization config
- metrics
- prompts/template version

This is essential if you want the project to look production-minded.

---

## Drift monitoring ideas
In finance, drift can be:
- new macroeconomic vocabulary
- new regulation
- market regime change
- sector-specific shifts

Track:
- label distribution drift
- embedding drift
- summary length drift
- out-of-domain ticker / company inputs

---

## Phase 5 – Deployment and MLOps

## Target architecture

### Inference pipeline
1. ingest document / transcript
2. preprocess + chunk
3. run summarization / classification
4. post-process JSON
5. run guardrails
6. log prediction and metadata
7. return API response

---

## Docker
Containerize:
- FastAPI or Flask app
- model loader
- tokenizer
- adapter
- inference pipeline
- metrics exporter

### Example app modules
- `app/main.py`
- `app/model.py`
- `app/schemas.py`
- `app/guardrails.py`
- `app/metrics.py`

---

## Kubernetes or lightweight alternative

### Lightweight local/demo options
- Docker Compose
- K3s
- Ray Serve
- vLLM for serving if model architecture/support fits your deployment path

### If you want full production-style story
Use Kubernetes with:
- Deployment
- HPA
- rolling updates
- ConfigMap / Secret
- Prometheus scraping

---

## Observability

Track:
- request latency
- tokens generated
- GPU memory
- OOM count
- error rate
- queue depth
- prediction distribution
- guardrail rejection rate

Tools:
- Prometheus
- Grafana
- OpenTelemetry traces
- Loki / ELK for logs

---

## Guardrails / “gardrails”

For finance, guardrails should include:

### Input guardrails
- PII detection
- prompt injection detection
- unsupported-instrument warning
- missing-context warning

### Output guardrails
- force JSON schema
- detect hallucinated figures
- confidence threshold for risk label
- banned advice templates:
  - “guaranteed return”
  - “certain profit”
- compliance disclaimer if needed

### Human-in-the-loop rule
If:
- confidence < threshold
- high-risk classification with weak evidence
- unsupported asset class
then route to manual review

---

## Versioning
Track:

### Model
- base model version
- adapter version
- quantization version
- merged/unmerged status

### Data
- raw dataset ID
- filtered dataset hash
- prompt generation version
- annotation guideline version

### Evaluation
- benchmark version
- scoring script version

---

# 7. Concrete experimental design for your portfolio

## Recommended final project structure

### Experiment A: Financial instruction tuning
- Base: `Qwen/Qwen2.5-3B-Instruct`
- Data: `sujet-ai/Sujet-Finance-Instruct-177k`
- Method: QLoRA 4-bit NF4
- Goal: improve finance-domain instruction responses

### Experiment B: Earnings-call risk summarization
- Base: same model
- Data: `Bose345/sp500_earnings_transcripts`
- Build custom summary/risk labels
- Goal: long-document domain adaptation

### Experiment C: Quantization tradeoff benchmark
Compare:
- BF16 base
- 8-bit
- 4-bit QLoRA
- optional GPTQ inference

### Experiment D: Cross-domain finance evaluation
Evaluate on:
- `FinancialPhraseBank`
- `Fin-RATE`
- custom risk test set

---

# 8. Recommended project deliverables

To make this portfolio project impressive, publish:

## 1. GitHub repo
Include:
- data cards
- training scripts
- evaluation scripts
- Dockerfile
- API server
- benchmark notebook

## 2. Technical report
Sections:
- motivation
- QLoRA theory
- finance dataset curation
- quantization experiments
- results
- limitations
- deployment plan

## 3. Colab notebooks
At minimum:
- notebook 1: data loading / preprocessing
- notebook 2: QLoRA fine-tuning
- notebook 3: quantized inference benchmark
- notebook 4: evaluation dashboard

## 4. Demo API / Streamlit
Input:
- earnings transcript / SEC section  
Output:
- summary
- risk label
- evidence
- confidence
- latency
- memory stats

---

# 9. Recommended implementation stack

## Training
- `transformers`
- `datasets`
- `peft`
- `trl`
- `accelerate`
- `bitsandbytes` ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

## Evaluation
- `evaluate`
- `scikit-learn`
- `bert-score`
- custom numeric/entity metrics

## Experiment tracking
- `wandb` or `mlflow`

## Serving
- `FastAPI`
- `uvicorn`
- `Docker`

## Monitoring
- Prometheus
- Grafana
- OpenTelemetry

---

# 10. Example repo layout

```text
finance-qlora-project/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ configs/
│  ├─ qlora_qwen3b.yaml
│  ├─ qlora_phi3.yaml
│  └─ eval.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ annotations/
├─ notebooks/
│  ├─ 01_load_finance_datasets.ipynb
│  ├─ 02_build_risk_labels.ipynb
│  ├─ 03_train_qlora.ipynb
│  ├─ 04_quant_benchmark.ipynb
│  └─ 05_eval_dashboard.ipynb
├─ src/
│  ├─ data/
│  │  ├─ load_hf.py
│  │  ├─ preprocess.py
│  │  └─ chunking.py
│  ├─ training/
│  │  ├─ train_sft.py
│  │  ├─ qlora_config.py
│  │  └─ prompts.py
│  ├─ eval/
│  │  ├─ summarize_metrics.py
│  │  ├─ classify_metrics.py
│  │  ├─ factuality.py
│  │  └─ numeric_checks.py
│  ├─ serving/
│  │  ├─ api.py
│  │  ├─ guardrails.py
│  │  └─ schemas.py
│  └─ utils/
├─ docker/
│  ├─ Dockerfile
│  └─ docker-compose.yml
└─ reports/
   └─ final_report.md
```

---

# 11. Best “story” for interviews / portfolio

Frame the project like this:

> I built a parameter-efficient finance-domain LLM adaptation pipeline using QLoRA on compact open models. I compared 16-bit, 8-bit, and 4-bit variants for earnings-call and SEC risk tasks, evaluated not only standard metrics but finance-specific factuality and numeric-faithfulness metrics, and designed a deployment-ready inference service with guardrails, monitoring, and version tracking.

That sounds strong because it covers:
- ML
- LLM engineering
- quantization
- domain adaptation
- evaluation
- MLOps

---

# 12. My final recommended setup

If you want the **best practical version** of this project:

## Base model
- `Qwen/Qwen2.5-3B-Instruct` ([huggingface.co](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct?utm_source=openai))

## Main training data
- `sujet-ai/Sujet-Finance-Instruct-177k` ([huggingface.co](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k?utm_source=openai))
- `Bose345/sp500_earnings_transcripts` ([huggingface.co](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts?utm_source=openai))

## Supplemental SEC corpus
- `nvidia/Nemotron-SpecializedDomains-Finance-v1` ([huggingface.co](https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1?utm_source=openai))

## Evaluation data
- `lmassaron/FinancialPhraseBank` ([huggingface.co](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank?utm_source=openai))
- `GGLabYale/Fin-RATE` ([huggingface.co](https://huggingface.co/datasets/GGLabYale/Fin-RATE?utm_source=openai))

## Training method
- 4-bit NF4 QLoRA with PEFT + bitsandbytes + TRL ([huggingface.co](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai))

## Core reference repo
- `artidoro/qlora` ([github.com](https://github.com/artidoro/qlora?utm_source=openai))

---

# 13. Important caveats

1. **Risk-level labels will likely need custom creation**  
   There isn’t a clean universal HF benchmark exactly matching `high / medium / low risk`.

2. **Finance instruction datasets can be noisy**  
   Curate carefully; do not train blindly on huge mixed corpora.

3. **Quantized training quality depends heavily on formatting and eval**  
   Better prompts + cleaner labels often matter more than tiny hyperparameter tweaks.

4. **For finance, factuality beats fluency**  
   Your project should emphasize numbers, entities, evidence, and calibration.

---

If you want, next I can turn this into one of these:

1. a **1-page shortlist of best datasets/models/repositories**, or  
2. a **full GitHub-ready project blueprint** with directory structure, scripts, and milestones, or  
3. a **working Colab-style notebook draft** for `Qwen2.5-3B + QLoRA + finance dataset`.

---
Learn more:
1. [Bose345/sp500\_earnings\_transcripts · Datasets at Hugging Face](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts?utm_source=openai)
2. [finosfoundation/EarningsCallTranscript · Datasets at Hugging Face](https://huggingface.co/datasets/finosfoundation/EarningsCallTranscript?utm_source=openai)
3. [JanosAudran/financial-reports-sec · Datasets at Hugging Face](https://huggingface.co/datasets/JanosAudran/financial-reports-sec?utm_source=openai)
4. [nvidia/Nemotron-SpecializedDomains-Finance-v1 · Datasets at Hugging Face](https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1?utm_source=openai)
5. [lmassaron/FinancialPhraseBank · Datasets at Hugging Face](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank?utm_source=openai)
6. [FinanceMTEB/financial\_phrasebank · Datasets at Hugging Face](https://huggingface.co/datasets/FinanceMTEB/financial_phrasebank?utm_source=openai)
7. [LLukas22/fiqa · Datasets at Hugging Face](https://huggingface.co/datasets/LLukas22/fiqa?utm_source=openai)
8. [NOSIBLE/financial-sentiment · Datasets at Hugging Face](https://huggingface.co/datasets/NOSIBLE/financial-sentiment?utm_source=openai)
9. [hongfenglu/finance-alpaca · Datasets at Hugging Face](https://huggingface.co/datasets/hongfenglu/finance-alpaca?utm_source=openai)
10. [sujet-ai/Sujet-Finance-Instruct-177k · Datasets at Hugging Face](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k?utm_source=openai)
11. [Josephgflowers/Finance-Instruct-500k · Datasets at Hugging Face](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k/viewer/default/train?utm_source=openai)
12. [GGLabYale/Fin-RATE · Datasets at Hugging Face](https://huggingface.co/datasets/GGLabYale/Fin-RATE?utm_source=openai)
13. [Financbase/financbase-10k-jsonl · Hugging Face](https://huggingface.co/Financbase/financbase-10k-jsonl?utm_source=openai)
14. [Magneto/qa\_with\_personas · Datasets at Hugging Face](https://huggingface.co/datasets/Magneto/qa_with_personas?utm_source=openai)
15. [Qwen/Qwen2.5-3B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct?utm_source=openai)
16. [microsoft/Phi-3.5-mini-instruct · Hugging Face](https://huggingface.co/microsoft/Phi-3.5-mini-instruct?utm_source=openai)
17. [qualcomm/Phi-3.5-mini-instruct · Hugging Face](https://huggingface.co/qualcomm/Phi-3.5-mini-instruct?utm_source=openai)
18. [google/gemma-2-2b-it · Hugging Face](https://huggingface.co/google/gemma-2-2b-it?utm_source=openai)
19. [meta-llama/Llama-3.2-1B-Instruct · Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct?utm_source=openai)
20. [mendeza/Llama-3.2-3B-Instruct · Hugging Face](https://huggingface.co/mendeza/Llama-3.2-3B-Instruct?utm_source=openai)
21. [ibm-granite/granite-3.1-2b-instruct · Hugging Face](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct?utm_source=openai)
22. [rapidfire-ai-inc/Mistral-7B-Instruct-v0.3-bnb-4bit · Hugging Face](https://huggingface.co/rapidfire-ai-inc/Mistral-7B-Instruct-v0.3-bnb-4bit?utm_source=openai)
23. [Quantization · Hugging Face](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=openai)
24. [PEFT Integration · Hugging Face](https://huggingface.co/docs/trl/v1.0.0/en/peft_integration?utm_source=openai)
25. [bitsandbytes · Hugging Face](https://huggingface.co/docs/transformers/v4.51.1/quantization/bitsandbytes?utm_source=openai)
26. [GitHub - artidoro/qlora: QLoRA: Efficient Finetuning of Quantized LLMs](https://github.com/artidoro/qlora?utm_source=openai)
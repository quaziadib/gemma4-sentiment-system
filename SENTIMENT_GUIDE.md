# Bangla Sentiment Analysis Quick Start

This project uses `gemma-4-26B-A4B` served locally through `llama.cpp` with an OpenAI-compatible endpoint. The sentiment pipeline classifies Bangla text into five labels:

- `0` `Strongly Negative`
- `1` `Weakly Negative`
- `2` `Neutral`
- `3` `Weakly Positive`
- `4` `Strongly Positive`

## 1. Start `llama-server`

Use the existing startup script. By default it listens on port `8000`:

```bash
bash start_server.sh
```

The inference script first tries:

```text
http://localhost:8001/v1/chat/completions
```

and then automatically falls back to:

```text
http://localhost:8000/v1/chat/completions
```

If your server is listening somewhere else, pass `--api_url` or set `LLAMA_API_URL`.

Example direct startup pattern for Gemma 4 GGUF with `llama.cpp`:

```bash
python3 -m llama_cpp.server \
  --model /path/to/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --n_gpu_layers -1 \
  --n_ctx 4096 \
  --n_batch 512 \
  --host 0.0.0.0 \
  --port 8001 \
  --chat_format gemma
```

## 2. Single Text Classification

```bash
python inference_sentiment.py --text "বাংলা টেক্সট"
```

Example:

```bash
python inference_sentiment.py --text "এই সেবাটা মোটামুটি ভালো, তবে আরও দ্রুত হলে ভালো হতো।"
```

## 3. Batch Processing

Input can be a plain text file with one sample per line, or a `.jsonl` file with a `text` field.

```bash
python inference_sentiment.py --input_file data.txt --output_file results.jsonl
```

Example JSONL input line:

```json
{"text": "পণ্যটি খারাপ না, ব্যবহার করা যায়।"}
```

Each output line is a JSON object like:

```json
{"input_text":"পণ্যটি খারাপ না, ব্যবহার করা যায়।","sentiment":"Weakly Positive","confidence":0.73,"label_id":3}
```

## 4. Evaluation

Prepare a `.jsonl` or `.csv` test file with:

- `text`: Bangla input text
- `label`: one of `Strongly Negative`, `Weakly Negative`, `Neutral`, `Weakly Positive`, `Strongly Positive`

Run:

```bash
python evaluate_sentiment.py --test_file test.jsonl
```

This prints:

- Accuracy
- Macro F1
- Weighted F1
- Per-class precision, recall, and F1
- Confusion matrix
- Cohen's kappa

It also saves a detailed JSON file with per-sample predictions and aggregate metrics.

## 5. Example Outputs By Class

Strongly Negative:

```json
{"input_text":"এই পণ্যটি একদম বাজে, টাকা নষ্ট হলো।","sentiment":"Strongly Negative","confidence":0.95,"label_id":0}
```

Weakly Negative:

```json
{"input_text":"সেবাটা তেমন ভালো না, আরও ভালো হতে পারত।","sentiment":"Weakly Negative","confidence":0.76,"label_id":1}
```

Neutral:

```json
{"input_text":"আজ বিকেল ৫টায় সভা শুরু হবে।","sentiment":"Neutral","confidence":0.93,"label_id":2}
```

Weakly Positive:

```json
{"input_text":"মন্দ না, মোটামুটি ভালোই লেগেছে।","sentiment":"Weakly Positive","confidence":0.79,"label_id":3}
```

Strongly Positive:

```json
{"input_text":"অসাধারণ অভিজ্ঞতা, সত্যিই দারুণ লেগেছে!","sentiment":"Strongly Positive","confidence":0.97,"label_id":4}
```

## 6. Notes

- The prompt is designed to handle sarcasm, idioms, mixed sentiment, and code-mixed Bangla-English text.
- The model must always choose exactly one label.
- `confidence` is the model's self-reported fit to the chosen class, constrained to `0.0` through `1.0`.

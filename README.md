# Rwandan Legal Assistant

An English question-answering assistant for Rwandan citizens. It answers questions about land, labour, persons and family, and the general penal code, using only the retrieved text of the official laws, and it cites the articles it relies on. When that text does not contain the answer, it refuses.

The primary dataset is the official English text of those laws, taken from the [Portal of Rwandan Laws and Case Laws](https://www.amategeko.gov.rw/). The statutes are consolidated, chunked, embedded, and stored for vector search. A smaller set of question-answer pairs is generated from that corpus so the model can be fine-tuned to cite articles and to refuse. Those pairs are instruction data derived from the laws. They are not the primary dataset.

- Live application: [https://qa-bot-fmwt.onrender.com/](https://qa-bot-fmwt.onrender.com/)
- Fine-tuned adapter: [kagaba-etienne/rwandan-legal-assistant-qlora](https://huggingface.co/kagaba-etienne/rwandan-legal-assistant-qlora), loaded on [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Law corpus: [https://www.amategeko.gov.rw/](https://www.amategeko.gov.rw/)
- Demo video: [https://youtu.be/iQSEPt51s_A](https://youtu.be/iQSEPt51s_A)

## Problem

Citizens need a practical way to ask what a current Rwandan law says, in everyday language, and to see which article the answer comes from. A general chatbot is a poor fit: the gazettes are long, bilingual, and amended over time, and an answer that is not tied to the consolidated text can invent a penalty, an office, or a deadline.

This project builds a domain assistant for four areas of domestic law:

- Law N° 27/2021 governing land
- Law N° 66/2018 regulating labour in Rwanda, as amended by Law N° 027/2023
- Law N° 71/2024 governing persons and family
- Law N° 68/2018 determining offences and penalties in general, as amended by Law N° 69/2019 and Law N° 059/2023

The assistant retrieves the closest articles, answers from that context, cites the article numbers, and uses one fixed refusal when the context is not enough: `I cannot answer this question based on the provided Rwandan legal context.`

## Dataset

The source documents are the official PDF gazettes in `data/raw/`. Each gazette prints Kinyarwanda, English, and French in parallel columns. Seven PDFs are used: the four base laws above and the three amending laws. The English column is the text the system indexes and cites.

After consolidation the corpus has **1,057 chunks** in [`data/processed/rwanda_law_corpus.jsonl`](data/processed/rwanda_law_corpus.jsonl):

- Governing land: 119
- Regulating labour in Rwanda: 155
- Governing persons and family: 434
- Determining offences and penalties in general: 349

Of those chunks, 945 are full articles and 112 are individual items split out of definition articles. Each chunk keeps a stable id, the law name, the article number, the text, and a citation such as `Article 158 of Law N° 68/2018`. Definition items also store the defined term. Other articles store the ids of definitions whose terms appear in the article.

The laws are appropriate for this task because they are the authoritative public text for the four domains, they are amended in place rather than replaced as a whole, and they are what a citizen would be expected to follow. The main limitations are the narrow domain (tax, immigration, company law, and similar areas are absent), English-only indexing of trilingual gazettes, and extraction noise from multi-column PDFs.

## Data preparation

```text
data/raw  →  data/interim/unedited  →  data/interim/edited  →  data/processed  →  MongoDB Atlas Vector Search
```

1. **Raw to unedited.** [`scripts/extract_english_columns.py`](scripts/extract_english_columns.py) opens each PDF with pdfplumber, crops the English column, drops the running header and footer, and writes one text file per law to `data/interim/unedited/`. Crop ratios and page ranges are set per gazette because the column positions are not identical.

2. **Manual edit.** The unedited files are copied to `data/interim/edited/` and cleaned by hand: page markers, headings that are not part of an article, and footers that survived the crop. Amendments were read one by one so each change could be recorded as a repeal, a replacement, or an insertion.

3. **Edited text to the corpus.** [`scripts/extract_articles.py`](scripts/extract_articles.py) splits each file on article headings, rejoins sentences broken by the PDF extraction, and applies the amendment map onto the base law. Repealed articles are removed, replaced articles are overwritten, and inserted articles are kept beside the article they follow. Long definition articles are split into one chunk per defined term. Each remaining article is tagged with the related definition chunks. The script writes `data/processed/rwanda_law_corpus.jsonl`.

4. **Corpus to vector search.** [`scripts/embed_and_index.py`](scripts/embed_and_index.py) embeds each chunk with OpenAI `text-embedding-3-small` (1,536 dimensions) and inserts the documents into MongoDB Atlas. The collection uses a vector search index named `vector_index` on the `embedding` field, with dot-product similarity. At question time the same embedding model retrieves the top four chunks.

## Instruction data for fine-tuning

The gazettes contain statutes, not citizen questions, so the corpus alone cannot teach the model the answer format. [`scripts/synthetic_qa_generation.py`](scripts/synthetic_qa_generation.py) builds instruction pairs from the indexed articles. `gpt-6-luna` reads a primary article and up to three neighbouring chunks and writes the pairs. This set is used only to fine-tune answer style, citation, and refusal. Retrieval at inference time still runs over the law corpus.

Positive pairs come from substantive articles, not definition chunks. For each article the script asks for:

- one direct question answerable from that article alone
- an optional second pair when a related definition is needed to interpret a term
- an optional third pair when another article adds a penalty, exception, or next step

Questions are written in the first person. Answers must begin from the citations in the supplied text and must not add facts that are not there.

Refusal pairs teach the model the fixed refusal sentence. One hundred and fifty are generated, in three categories:

- **A.** A real legal question outside the four laws, such as tax or immigration.
- **B.** A non-legal everyday question.
- **C.** A question about the supplied article that asks for a detail the article does not state, such as a specific office or a deadline the text leaves open.

Category A and B questions are seeded from rotating topic lists so the generator does not repeat one refusal scenario. For every refusal, the question is embedded and the chunks closest to that question are saved as the context, which is the same retrieval setup the application uses.

The full instruction file is `data/processed/qlora_synthetic_dataset.jsonl` (1,831 positive pairs and 150 refusals). A stratified 90/10 split, shuffled inside each class with seed 42, writes:

- `data/processed/qlora_train.jsonl`: 1,782 examples (1,647 positive, 135 refusals)
- `data/processed/qlora_val.jsonl`: 199 examples (184 positive, 15 refusals)

The generator can echo the wording and the gaps of the source articles, and the validation questions are drawn from the same process as the training questions. Scores on this split measure citation, faithfulness to the retrieved text, and refusal behaviour. They are not a measure of performance on questions written independently of the corpus.

## Model

The assistant is a QLoRA adapter on `Qwen/Qwen2.5-0.5B-Instruct`. Training is in [`notebooks/finetuning.ipynb`](notebooks/finetuning.ipynb).

The base model is loaded in 4-bit NormalFloat (NF4) with bitsandbytes (`BitsAndBytesConfig`: `load_in_4bit`, `bnb_4bit_quant_type="nf4"`). A LoRA adapter is then trained with Hugging Face PEFT and the Transformers `Trainer`:

- rank 32, alpha 32, dropout 0.05, no bias
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- 17,596,416 trainable parameters out of 511,629,184 (3.44%)
- 3 epochs, learning rate 2e-4, warmup 50 steps
- batch size 1 with gradient accumulation 4
- optimizer `paged_adamw_8bit`, gradient checkpointing, maximum sequence length 2048
- evaluation and checkpoints every 100 steps, best checkpoint restored at the end

The run used one RTX 5090 on a RunPod pod and finished in about 25 minutes (1,338 steps). The best validation loss was 0.210, at the final checkpoint. The notebook pushes the adapter to [kagaba-etienne/rwandan-legal-assistant-qlora](https://huggingface.co/kagaba-etienne/rwandan-legal-assistant-qlora). Local checkpoints from that run are under `model/`.

The saved notebook points `TRAIN_PATH` and `VAL_PATH` at the files used on the training machine. To repeat the run, point those two constants at `data/processed/qlora_train.jsonl` and `data/processed/qlora_val.jsonl`.

## Experiments and results

The baseline is the same `Qwen2.5-0.5B-Instruct` checkpoint with no adapter. Both the baseline and the fine-tuned adapter are scored on the 199 validation examples under two retrieval settings, so the comparison separates the effect of the adapter from the effect of showing the model the articles. Predictions are generated on the RunPod endpoint and written under `data/results/`:

- `baseline_no_rag` and `finetuned_no_rag` receive the question and an empty context. The correct behaviour on every row is the fixed refusal, because no legal text was provided.
- `baseline_with_rag` and `finetuned_with_rag` receive the same retrieved context stored with the validation example. This is the setting the web application uses. The deployed system is `finetuned_with_rag`.

Automatic scores use two kinds of metric. Refusal exact match is 1 when the output is exactly the mandated refusal sentence. False refusal is 1 when the model emits that sentence on a question the context can answer. Faithfulness, answer relevance, and citation are 1–5 G-Eval scores: `gpt-4o` rates the response against the context using fixed criteria, and the score is the probability-weighted value of the digit it assigns. The gold answer is not shown to the judge. Citation is scored on positive examples with retrieval, where an article number is available to cite.

With retrieval, on 199 validation examples (184 answerable, 15 refusals):

- Fine-tuned: citation 3.69, faithfulness 4.31, relevance 4.15, refusal exact match 0.80, false refusal 0.00
- Baseline: citation 3.38, faithfulness 3.84, relevance 4.02, refusal exact match 0.00, false refusal 0.00

The adapter improves citation and faithfulness, and it is the only system that produces the required refusal. It does not refuse answerable questions in this split. Refusal quality depends on the category. With retrieval, exact-match refusal is 1.00 for out-of-scope law (A, n=6) and for non-legal questions (B, n=4), and 0.40 for missing-detail questions (C, n=5). Category C is the hard case: the retrieved article is on the right topic, and the question asks for a procedure or figure the article never states. The baseline, given the same context, answers in prose and never emits the refusal sentence.

Without retrieval, where every question should be refused:

- Fine-tuned: refusal exact match 0.24, faithfulness 1.97, relevance 1.94
- Baseline: refusal exact match 0.00, faithfulness 1.03, relevance 1.34

The adapter refuses more often than the baseline when no articles are attached, and it still answers most of those questions from parametric memory. The application therefore always retrieves before it calls the model. The four full summaries are the `summary.csv` files in `data/results/`. [`data/results/human_evaluation/review.csv`](data/results/human_evaluation/review.csv) places the four systems' answers beside the question, the context, and the gold response for qualitative review.

## Application

The chat page is a React Router application in `frontend/`. In development, Vite proxies `/api` to the API. The API is FastAPI (`src/main.py`). `POST /api/chat` accepts `{ "message": "..." }`.

[`src/services/chat.py`](src/services/chat.py) embeds the question, runs a MongoDB vector search for the top four chunks, and formats them as numbered documents with citations. [`src/lib/qwen.py`](src/lib/qwen.py) sends that prompt to a RunPod serverless endpoint that serves the base model with the Hugging Face adapter loaded. The endpoint is called through its OpenAI-compatible chat route. The adapter location configured on the endpoint is [kagaba-etienne/rwandan-legal-assistant-qlora](https://huggingface.co/kagaba-etienne/rwandan-legal-assistant-qlora).

The browser never talks to RunPod or MongoDB directly. The Render deployment is one container: the root [`Dockerfile`](Dockerfile) builds the React Router client in a Node stage, then a Python stage copies that static build and serves it from FastAPI on port 8000. The same process can be started as two programs during development.

## Running the system

Create a `.env` file in the repository root:

```bash
OPENAI_API_KEY=
MONGO_URI=
RUNPOD_API_KEY=
RUNPOD_ENDPOINT_ID=
# optional; defaults are rwandan_law and articles
DB_NAME=rwandan_law
COLLECTION_NAME=articles
```

`OPENAI_API_KEY` is used for embeddings and for G-Eval. `MONGO_URI` is the Atlas cluster that holds the article index. `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` select the serverless endpoint. The API process reads these in [`src/config/env.py`](src/config/env.py). Python dependencies are in the [`Pipfile`](Pipfile) (Python 3.14). Frontend dependencies are in [`frontend/package.json`](frontend/package.json).

### Docker

This is the image deployed on Render. The frontend is built inside the image and served by FastAPI.

```bash
docker build -t qa-bot .
docker run --env-file .env -p 8000:8000 qa-bot
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

### Frontend and API as separate processes

```bash
pipenv install
pipenv run api
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server is at [http://127.0.0.1:5173/](http://127.0.0.1:5173/) and forwards `/api` to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

### Data preparation, training, and evaluation

From the repository root, with the `.env` file in place and the PDFs in `data/raw/`:

```bash
pipenv install
pipenv run python scripts/extract_english_columns.py
# review data/interim/unedited and place the cleaned files in data/interim/edited
pipenv run python scripts/extract_articles.py
pipenv run embed_and_index
pipenv run generate_synthetic_qa_dataset
```

Fine-tune by running [`notebooks/finetuning.ipynb`](notebooks/finetuning.ipynb) on a GPU machine after pointing it at the train and validation JSONL files. Then generate validation predictions through the RunPod endpoint and score them:

```bash
pipenv run predict
```

`predict` writes `predictions.jsonl` for all four experiment folders and scores them. Pass `--no-score` to write the prediction files only. `pipenv run evaluate --all` scores prediction files that are already on disk. Pass `--variant finetuned_with_rag` (or `baseline_with_rag`, `finetuned_no_rag`, `baseline_no_rag`) to either command to run a single condition.

## Repository layout

```text
data/raw/                  official law PDFs
data/interim/unedited/     English column extracted from each PDF
data/interim/edited/       manually cleaned text used for article extraction
data/processed/            law corpus and instruction-tuning splits
data/results/              predictions, G-Eval summaries, qualitative review sheet
notebooks/finetuning.ipynb QLoRA training
model/                     local adapter checkpoints from the training run
scripts/                   extraction, indexing, instruction data, prediction, evaluation
src/                       FastAPI application, retrieval, and RunPod client
frontend/                  React Router chat interface
Dockerfile                 multi-stage image used on Render
```

## Acknowledgements

- Law texts from the [Portal of Rwandan Laws and Case Laws](https://www.amategeko.gov.rw/).
- Base model: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- QLoRA and 4-bit training via bitsandbytes, Hugging Face Transformers, and PEFT. Dettmers et al., [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314).
- G-Eval for faithfulness, relevance, and citation. Liu et al., [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634).
- Embeddings and vector search: OpenAI `text-embedding-3-small` and MongoDB Atlas Vector Search.
- Application stack: FastAPI, React Router, and the libraries listed in the [`Pipfile`](Pipfile) and [`frontend/package.json`](frontend/package.json).

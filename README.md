# Fine-tuning a small model for HRCS health categories

This repository trains a small open model, Qwen3-0.6B, to suggest HRCS health categories for UK health research grants. It also compares the model with Claude and with a simple keyword model. The trained model is on Hugging Face at [rsg07/qwen3-0.6b-hrcs-health-categories](https://huggingface.co/rsg07/qwen3-0.6b-hrcs-health-categories).

HRCS is the system UK funders use to record which area of health a grant is about. Each grant gets one to five of 21 categories. The model scored 0.818 on the official test set of 972 grants, 2.8 points higher than Claude Sonnet 5.5. The score is micro-F1, an accuracy measure from 0 to 1.

You do not need this repository to use the trained model. The Hugging Face page has the code for that.

## What you need

- A Mac with Apple silicon. Training and prediction run on the Mac's GPU.
- [uv](https://docs.astral.sh/uv/), a tool that installs the exact library versions this project used. It also installs Python 3.12 if you don't have it.

## Install

```bash
git clone https://github.com/rsengupta94/qwen3-0.6b-classification-finetuning.git
cd qwen3-0.6b-classification-finetuning
uv sync
```

`uv sync` sets up a local environment with the library versions listed in `uv.lock`.

## Train and score the model

Run these commands in order, from the repository folder.

```bash
uv run python download_data.py
```

This downloads the HRCS data from Hugging Face into `data/raw/`.

```bash
uv run python check_data.py
```

This step is optional. It prints checks on the raw data and changes no files.

```bash
uv run python prepare_data.py
```

This cleans the text and splits the grants into training, validation and test sets in `data/splits/`. The validation set is used for choices during training. The test set is used once, for the final score.

```bash
uv run python tokenize_data.py
```

This turns each grant into tokens, the small pieces of text the model reads, and saves them in `data/tokenized/`.

```bash
uv run python train.py smoke --test
```

This is a five-minute check that every part of training works.

```bash
uv run python train.py run1
```

This trains the model and saves it in `runs/run1/`. It took about 10 hours on an Apple M3 Pro with 36 GB of memory.

```bash
uv run python predict.py run1 --test
```

This uses the validation set to choose the cut-off score at which the model suggests a category. It then saves predictions for the validation and test sets in `data/predictions/`.

```bash
uv run python score.py data/predictions/test/run1.csv
```

This prints the test scores. Your scores may differ slightly from the ones above.

## Compare with other models

These steps are optional.

```bash
uv run python keyword_model.py
uv run python keyword_model.py --test
```

The keyword model looks only at which words a grant uses. The first command trains it and scores it on the validation set. The second saves its test predictions.

```bash
uv run python fetch_guide.py
```

This downloads the official HRCS coding guide into a folder named `data/guide/` plus today's date. Claude reads this guide in its prompt. After the download, set `GUIDE_FILE` in `settings.py` to the new folder.

```bash
uv run python claude_baseline.py haiku fixed test
```

This asks Claude Haiku about every test grant, with the guide and five example grants. It needs [Claude Code](https://claude.com/claude-code) installed and signed in, and it uses your Claude account. Write `sonnet` in place of `haiku` to use Claude Sonnet. Write `zeroshot` in place of `fixed` to give Claude the guide without the examples.

## Look at the results

```bash
uv run python error_analysis.py run1
```

This shows where the model goes wrong on the test set. It also prints 25 wrong predictions from the validation set to read.

```bash
uv run python make_charts.py
```

This redraws the two charts on the Hugging Face page and saves them in `model_card/assets/`. It needs test predictions from the model, the keyword model and three Claude runs. These are Sonnet with examples, Haiku with examples and Haiku with the guide only.

## Settings

All shared values, such as file paths, the base model and the random seed, are in `settings.py`. Change them there, not in the scripts. The project needs no API keys.

## Status and license

The project is finished, and the model will not be trained again. The code is under the [MIT License](LICENSE). The trained model is under CC BY-NC 4.0, for non-commercial use only. The HRCS data is not stored in this repository. The HRCS categories belong to the UK Clinical Research Collaboration, NIHR publishes the dataset, and `download_data.py` fetches it.

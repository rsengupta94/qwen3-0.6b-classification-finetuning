---
license: cc-by-nc-4.0
language:
- en
library_name: transformers
pipeline_tag: text-classification
base_model: Qwen/Qwen3-0.6B-Base
datasets:
- NIHRDataInsights/HRCSData
tags:
- hrcs
- health-research
- research-funding
- multi-label-classification
- biomedical
model-index:
- name: qwen3-0.6b-hrcs-health-categories
  results:
  - task:
      type: text-classification
      name: Multi-label classification into 21 HRCS health categories
    dataset:
      type: NIHRDataInsights/HRCSData
      name: HRCS Data (official test file, 972 grants)
      split: test
    metrics:
    - type: f1
      name: Micro-F1 (threshold 0.435)
      value: 0.818
    - type: precision
      name: Micro precision
      value: 0.833
    - type: recall
      name: Micro recall
      value: 0.803
    - type: f1
      name: Macro-F1
      value: 0.759
---

# Qwen3-0.6B fine-tuned for HRCS health categories

This model suggests HRCS health categories for a UK health research grant. It reads the grant's title and abstract, and a trained coder then checks its suggestions. This project tests whether a small model, fine-tuned for one narrow task, can beat leading hosted models at that task. On HRCS coding it did, scoring 2.8 points higher than Claude Sonnet 5.5.

UK research funders use the Health Research Classification System (HRCS) to record which area of health a grant is about. Each grant gets one to five of 21 health categories, such as Cancer, Infection or Mental Health.

## Results

These are scores on the official HRCS test set of 972 grants, which the model did not see in training. Micro-F1 is the main score. It combines precision, the share of suggestions that are correct, with recall, the share of correct categories that the model finds.

<img class="dark:hidden" src="https://huggingface.co/rsg07/qwen3-0.6b-hrcs-health-categories/resolve/main/assets/results-light.png" alt="Micro-F1 for each row, and this model's lead over each row with its 95% range."/>
<img class="hidden dark:block" src="https://huggingface.co/rsg07/qwen3-0.6b-hrcs-health-categories/resolve/main/assets/results-dark.png" alt="Micro-F1 for each row, and this model's lead over each row with its 95% range."/>

| | Micro-F1 | Precision | Recall |
|---|---|---|---|
| **This model** | **0.818** | 0.833 | 0.803 |
| Keyword model | 0.808 | 0.795 | 0.822 |
| Claude Sonnet 5.5, HRCS guide and 5 examples | 0.789 | 0.805 | 0.775 |
| Claude Haiku 5.5, HRCS guide and 5 examples | 0.772 | 0.774 | 0.770 |
| Claude Haiku 5.5, HRCS guide only | 0.763 | 0.731 | 0.798 |
| NIHR's model | 0.81 | 0.83 | 0.80 |

The model scored 2.8 points higher than Claude Sonnet 5.5 and 4.6 points higher than Claude Haiku 5.5. A simple keyword model came within 1 point of it. When the categories depend mostly on the words a text uses, a keyword model can match a fine-tuned one at a fraction of the cost. NIHR's numbers come from [its model page](https://huggingface.co/NIHRDataInsights/HRCSHealthCategories), which does not say which test set it used.

## How to use

The input is the title, a new line, and then the abstract. The model gives each category a score from 0 to 1 and suggests the categories that score 0.435 or more. Every grant gets at least one category and at most five. The code below applies these rules.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "rsg07/qwen3-0.6b-hrcs-health-categories"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

def suggest(title, abstract, threshold=0.435):
    inputs = tok(f"{title}\n{abstract}", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        scores = torch.sigmoid(model(**inputs).logits)[0]
    picked = [i for i, s in enumerate(scores) if s >= threshold] or [int(scores.argmax())]
    picked = sorted(picked, key=lambda i: -scores[i])[:5]
    return [(model.config.id2label[i], round(float(scores[i]), 2)) for i in picked]

print(suggest("Grant title", "Grant abstract"))
```

The output looks like `[('infection', 0.98), ('blood', 0.63)]`. The first run downloads 2.2 GB. On an Apple M3 Pro CPU, one grant takes about half a second.

## Results by category

The model beats Claude Sonnet on most of the large categories, and on Stroke by the widest margin. Claude does better on three small ones: Injuries and Accidents, Eye, and Blood.

<img class="dark:hidden" src="https://huggingface.co/rsg07/qwen3-0.6b-hrcs-health-categories/resolve/main/assets/per-category-light.png" alt="F1 for each health category, this model and Claude Sonnet 5.5."/>
<img class="hidden dark:block" src="https://huggingface.co/rsg07/qwen3-0.6b-hrcs-health-categories/resolve/main/assets/per-category-dark.png" alt="F1 for each health category, this model and Claude Sonnet 5.5."/>

<details>
<summary>Scores for each category</summary>

| Category | Test grants | This model F1 | Claude Sonnet F1 |
|---|---|---|---|
| Generic health relevance | 255 | 0.81 | 0.77 |
| Cancer and neoplasms | 198 | 0.91 | 0.90 |
| Infection | 153 | 0.88 | 0.90 |
| Mental health | 113 | 0.83 | 0.79 |
| Neurological | 112 | 0.81 | 0.81 |
| Inflammatory and immune system | 65 | 0.65 | 0.63 |
| Metabolic and endocrine | 55 | 0.84 | 0.78 |
| Cardiovascular | 47 | 0.71 | 0.62 |
| Musculoskeletal | 43 | 0.82 | 0.76 |
| Oral and gastrointestinal | 43 | 0.74 | 0.64 |
| Reproductive health and childbirth | 41 | 0.85 | 0.81 |
| Stroke | 33 | 0.82 | 0.62 |
| Respiratory | 20 | 0.76 | 0.77 |
| Injuries and accidents | 17 | 0.71 | 0.88 |
| Blood | 15 | 0.62 | 0.74 |
| Eye | 14 | 0.80 | 0.90 |
| Renal and urogenital | 9 | 0.82 | 0.86 |

Four categories have too few test grants to score: Congenital Disorders, Skin, Ear, and Disputed Aetiology and Other.

</details>

## Model details

| | |
|---|---|
| Developed by | [Rajarshi Sengupta](https://github.com/rsengupta94) |
| Base model | [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), with all weights trained |
| Training data | [NIHRDataInsights/HRCSData](https://huggingface.co/datasets/NIHRDataInsights/HRCSData), UK grants from 2014 to 2022, with 19,175 for training and 1,990 for validation |
| Input | The first 512 tokens of the title and abstract |
| Threshold | 0.435, chosen on validation grants to find 80% of the correct categories |
| Training | 3 passes over the data |
| Code | [github.com/rsengupta94/qwen3-0.6b-classification-finetuning](https://github.com/rsengupta94/qwen3-0.6b-classification-finetuning) |

## License and credits

This model is for non-commercial use only, under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). HRCS belongs to the UK Clinical Research Collaboration ([hrcsonline.net](https://hrcsonline.net)). NIHR publishes the dataset. Qwen3-0.6B-Base is by the Qwen team, under Apache 2.0.

Questions are welcome in the Community tab or on [LinkedIn](https://www.linkedin.com/in/rajarshi-sengupta/).

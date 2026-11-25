# RedoxPotNet

RedoxPotNet is a transcriptomic scoring framework for estimating a redox-associated score from gene-expression profiles.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/XiaoJ327/RedoxPotNet.git
cd RedoxPotNet
pip install -e .
```

The complete gene sets and pretrained model weights are not included in this prepublication repository and will be made available upon publication of the associated manuscript. 

## Input format

For file-based prediction, the expression matrix must have **genes in rows** and **samples in columns**. The first column contains HGNC gene symbols.

```text
gene_name    Sample_1    Sample_2    Sample_3
A1BG         3.52        4.10        3.88
A2M          0.00        0.14        0.00
...
```

TPM can be used directly. RedoxPotNet converts expression values to within-sample ranks using the fixed background-gene universe bundled with the model.

## Python usage

```python
from redoxpotnet import RedoxPotNet

model = RedoxPotNet()
result = model.predict_file("expression_TPM.tsv")
result.to_csv("RedoxPotNet_scores.csv", index=False)
```

The output contains one score per sample together with input-gene coverage metrics.

## Command-line usage

```bash
python scripts/predict.py \
    --input examples/example_expression_TPM.tsv \
    --output RedoxPotNet_scores.csv

## Tested environment

RedoxPotNet has been tested with:

- Python 3.13.2
- PyTorch 2.6.0+cu124
- NumPy 2.1.3
- pandas 2.2.3
```

GPU acceleration is not required for RedoxPotNet inference.




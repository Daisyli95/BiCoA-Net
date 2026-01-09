# BiCoA-Net

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**BiCoA-Net: An Interpretable Bidirectional Co-Attention Framework for Predicting Protein-Ligand Binding Kinetics**

Official implementation of BiCoA-Net, a deep learning framework for predicting protein-ligand binding kinetics (pKoff) with enhanced interpretability through bidirectional co-attention mechanisms.

> **Status:** Under Review

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Running Predictions](#running-predictions)
- [Input Data Format](#-input-data-format)
- [Model Weights](#-model-weights)
- [License](#-license)

---

## 🔬 Overview

BiCoA-Net predicts protein-ligand binding kinetics with state-of-the-art performance and interpretability. The model uses bidirectional co-attention mechanisms to capture interactions between protein sequences and ligand structures.

**Key Applications:**
- Drug discovery and development
- Lead optimization
- Compound screening and prioritization

---

## ✨ Key Features

- **Easy to Use:** Simple command-line interface for predictions
- **Comprehensive Analysis:** Built-in tools for drug discovery metrics
- **Fast Inference:** GPU-accelerated predictions
- **Multiple Formats:** Supports both CSV and Excel input/output

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (for GPU support, optional but recommended)
- 8GB+ RAM (16GB+ recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Daisyli95/BiCoA-Net.git
cd BiCoA-Net
```

### Step 2: Install PyTorch
For **CUDA 11.8** (recommended for GPU acceleration):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For **CPU-only** installation:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> **Note:** For other CUDA versions, visit [PyTorch installation guide](https://pytorch.org/get-started/locally/)

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### 1. Download Model Weights
Download the pre-trained model checkpoint:
- [model.pt]([link-to-huggingface](https://huggingface.co/Daisyli95/BiCoA-Net/blob/main/model.pt)) (recommended)

Place the checkpoint in your working directory.

### 2. Prepare Your Data
Create a CSV or Excel file with the following columns:
- `FASTA`: Protein sequence in FASTA format
- `smiles`: Ligand structure in SMILES format

Example (`input.csv`):
```csv
FASTA,smiles
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL,CC(C)Cc1ccc(cc1)C(C)C(=O)O
```

### 3. Run Predictions
```bash
python inference.py --checkpoint model.pt --input input.csv
```

That's it! Your predictions will be saved in the `./predictions` directory.

---

## 📖 Usage

### Running Predictions

#### Basic Usage
```bash
python inference.py --checkpoint model.pt --input data.csv
```

#### Advanced Options
```bash
python inference.py \
    --checkpoint model.pt \
    --input file1.csv \
    --output-dir ./my_predictions \
    --device cuda \
    --batch-size 64
```

#### Command-Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--checkpoint` | Path to model checkpoint (.pt file) | - | ✅ |
| `--input` | Input data file(s) (.csv) | - | ✅ |
| `--output-dir` | Directory for saving results | `./predictions` | ❌ |
| `--device` | Computation device (`cuda` or `cpu`) | `cuda` | ❌ |
| `--batch-size` | Batch size for inference | `32` | ❌ |

#### Example: Processing Multiple Files
```bash
python inference.py \
    --checkpoint model.pt \
    --input dataset.csv \
    --output-dir ./batch_predictions
```

---

## 📊 Input Data Format

### Required Columns

Your input file must contain these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `FASTA` | string | Protein sequence in FASTA format | `MKTAYIAKQRQISFVK...` |
| `smiles` | string | Ligand structure in SMILES notation | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |

### Optional Columns

You can include additional columns for reference:
- `protein_name`: Protein identifier
- `ligand_name`: Compound identifier
- `experimental_pKoff`: Known experimental values (for validation)

### Supported File Formats
- **CSV** (`.csv`): Comma-separated values

### Example Input Files

**Example 1: Minimal format**
```csv
FASTA,smiles
MKTAYIAKQRQISFVK...,CC(C)Cc1ccc(cc1)C(C)C(=O)O
MASGADSKGD...,COc1ccc2c(c1)ncc(n2)CN
```


---
- Same data in Excel format with formatting

## 💾 Model Weights

Pre-trained models are available via Hugging Face: https://huggingface.co/Daisyli95/BiCoA-Net/tree/main . Please download the model.pt file.



### Using Custom Models

Train your own model and use it for inference:
```bash
python inference.py --checkpoint your_custom_model.pt --input data.csv
```
---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by the BiCoA-Net team**

[⭐ Star this repo](https://github.com/Daisyli95/BiCoA-Net) | [🐛 Report Bug](https://github.com/Daisyli95/BiCoA-Net/issues) | [💡 Request Feature](https://github.com/Daisyli95/BiCoA-Net/issues)

</div>

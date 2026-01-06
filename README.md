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
  - [Analyzing Results](#analyzing-results)
- [Input Data Format](#-input-data-format)
- [Output Files](#-output-files)
- [Model Weights](#-model-weights)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🔬 Overview

BiCoA-Net predicts protein-ligand binding kinetics with state-of-the-art performance and interpretability. The model uses bidirectional co-attention mechanisms to capture interactions between protein sequences and ligand structures.

**Key Applications:**
- Drug discovery and development
- Lead optimization
- Compound screening and prioritization
- Structure-activity relationship (SAR) studies

---

## ✨ Key Features

- **High Accuracy:** State-of-the-art performance on binding kinetics prediction
- **Interpretable:** Bidirectional co-attention provides insights into molecular interactions
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
- [model.pt](link-to-huggingface) (recommended)

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
python inference.py --checkpoint model.pt --input data.xlsx
```

#### Advanced Options
```bash
python inference.py \
    --checkpoint model.pt \
    --input file1.xlsx file2.csv \
    --output-dir ./my_predictions \
    --device cuda \
    --batch-size 64
```

#### Command-Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--checkpoint` | Path to model checkpoint (.pt file) | - | ✅ |
| `--input` | Input data file(s) (.csv or .xlsx) | - | ✅ |
| `--output-dir` | Directory for saving results | `./predictions` | ❌ |
| `--device` | Computation device (`cuda` or `cpu`) | `cuda` | ❌ |
| `--batch-size` | Batch size for inference | `32` | ❌ |

#### Example: Processing Multiple Files
```bash
python inference.py \
    --checkpoint model.pt \
    --input dataset1.xlsx dataset2.xlsx dataset3.csv \
    --output-dir ./batch_predictions
```

---

### Analyzing Results

After generating predictions, analyze them with drug discovery metrics:

```bash
python analyze.py --predictions-dir ./predictions
```

#### Analysis Options
```bash
python analyze.py \
    --predictions-dir ./predictions \
    --output-dir ./analysis \
    --outlier-method iqr \
    --outlier-threshold 1.5
```

#### Analysis Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--predictions-dir` | Directory with prediction files | - | - |
| `--output-dir` | Output directory | `./analysis_results` | - |
| `--outlier-method` | Outlier detection method | `iqr` | `iqr`, `zscore`, `none` |
| `--outlier-threshold` | Outlier threshold | `1.5` | any float |

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
- **Excel** (`.xlsx`): Microsoft Excel format

### Example Input Files

**Example 1: Minimal format**
```csv
FASTA,smiles
MKTAYIAKQRQISFVK...,CC(C)Cc1ccc(cc1)C(C)C(=O)O
MASGADSKGD...,COc1ccc2c(c1)ncc(n2)CN
```

**Example 2: With metadata**
```csv
protein_name,ligand_name,FASTA,smiles,experimental_pKoff
EGFR,Ibuprofen,MKTAYIAKQRQISFVK...,CC(C)Cc1ccc(cc1)C(C)C(=O)O,6.2
HSP90,Compound_123,MASGADSKGD...,COc1ccc2c(c1)ncc(n2)CN,7.1
```

---

## 📁 Output Files

### Prediction Files

For each input file, two output files are generated:

**predictions_{filename}.csv**
```csv
FASTA,smiles,predicted_pKoff
MKTAYIAKQRQISFVK...,CC(C)Cc1ccc(cc1)C(C)C(=O)O,6.45
```

**predictions_{filename}.xlsx**
- Same data in Excel format with formatting

### Analysis Files (when running `analyze.py`)

#### 📊 Visualizations
- `performance_dashboard.png`
  - 8-panel comprehensive analysis
  - Concordance Index as primary metric
  - Ranking performance and enrichment
  - Scatter plots and distributions

- `target_comparison.png`
  - Per-target performance breakdown
  - CI and correlation metrics
  - Sample distributions

#### 📋 Reports
- `comprehensive_report.txt`
  - Detailed metrics with drug discovery context
  - Clinical recommendations
  - Statistical analysis

#### 📁 Data Files
- `clean_predictions.csv/xlsx`: Cleaned prediction data
- `outliers.csv/xlsx`: Detected outliers (if any)

---

## 🎯 Drug Discovery Metrics

The analysis tool provides key metrics for drug discovery:

### Concordance Index (CI)
**Primary metric for compound ranking**
- CI ≥ 0.7: Excellent (suitable for lead optimization)
- CI ≥ 0.6: Good (useful for compound screening)
- CI < 0.6: Poor ranking ability

### Top-K Accuracy
Measures ability to identify top-performing compounds:
- Top-1, Top-5, Top-10 accuracy
- Critical for hit identification

### Enrichment Factor
Quantifies screening efficiency:
- Higher values indicate better compound prioritization
- Important for high-throughput screening campaigns

### Correlation Metrics
- **Pearson r**: Linear relationship strength
- **Spearman ρ**: Monotonic relationship strength
- **R² Score**: Variance explained by predictions

---

## 💾 Model Weights

Pre-trained models are available via Hugging Face:

| Model | Dataset | Performance | Download |
|-------|---------|-------------|----------|
| BiCoA-Net v1.0 | KinetX | CI: 0.78, R²: 0.72 | [Download](link) |

### Using Custom Models

Train your own model and use it for inference:
```bash
python inference.py --checkpoint your_custom_model.pt --input data.csv
```

---

## 📚 Citation

If you use BiCoA-Net in your research, please cite:

```bibtex
@article{bicoanet2024,
  title={BiCoA-Net: An Interpretable Bidirectional Co-Attention Framework for Predicting Protein-Ligand Binding Kinetics},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  note={Under Review}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs via [Issues](https://github.com/Daisyli95/BiCoA-Net/issues)
- Submit feature requests
- Create pull requests

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **GitHub Issues:** [Create an issue](https://github.com/Daisyli95/BiCoA-Net/issues)
- **Email:** [your-email@example.com]

---

## 🙏 Acknowledgments

We thank the contributors to the following projects:
- PyTorch team for the deep learning framework
- RDKit for cheminformatics tools
- The scientific community for valuable feedback

---

## 📝 Changelog

### Version 1.0.0 (Current)
- Initial release
- Core prediction functionality
- Comprehensive analysis tools
- Publication-quality visualizations

---

<div align="center">

**Made with ❤️ by the BiCoA-Net team**

[⭐ Star this repo](https://github.com/Daisyli95/BiCoA-Net) | [🐛 Report Bug](https://github.com/Daisyli95/BiCoA-Net/issues) | [💡 Request Feature](https://github.com/Daisyli95/BiCoA-Net/issues)

</div>

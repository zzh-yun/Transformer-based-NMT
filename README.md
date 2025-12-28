# Transformer-based Neural Machine Translation (NMT)

A Chinese-English neural machine translation system based on the Transformer architecture, supporting both training from scratch and fine-tuning of pre-trained models.

🚀🚀🚀 Due to the model checkpoint is too big to put on github, so the checkpoint is upload to the Huggingface: https://huggingface.co/DarcyCheng/Transformer-based-NMT


## Introduction

This project implements a complete Transformer-based NMT system, with core tasks including:

### 1. From Scratch Training
Construct and train a Chinese-English translation model based on the Transformer architecture, including:
- Adoption of the Encoder-Decoder structure
- Model training from scratch
- Support for distributed training (multi-GPU)

### 2. Architectural Ablation
Implement and compare different architectural variants:
- **Positional Encoding Schemes**: Absolute Positional Encoding vs Relative Positional Encoding
- **Normalization Methods**: LayerNorm vs RMSNorm

### 3. Hyperparameter Sensitivity
Evaluate the impact of hyperparameter adjustments on translation performance:
- Batch Size
- Learning Rate
- Model Scale

### 4. From Pretrained Language Model
Support fine-tuning based on pre-trained language models (e.g., T5) and compare performance with models trained from scratch.

## Data Preparation

The dataset consists of four JSONL files, corresponding to:
- **Small Training Set**: 100k samples
- **Large Training Set**: 10k samples
- **Validation Set**: 500 samples
- **Test Set**: 200 samples

Each line in the JSONL files contains a parallel Chinese-English sentence pair. The final model performance will be evaluated based on the results from the test set.

Data Format Example:
```json
{"zh": "你好，世界。", "en": "Hello, world."}
```

## Environment

### System Requirements
- **Python**: 3.9.25
- **PyTorch**: 2.0.1+cu118
- **CUDA**: 11.8 (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch>=2.0.1`
- `torchvision`
- `numpy`
- `matplotlib`
- `tqdm`
- `hydra-core`
- `omegaconf`
- `sentencepiece`
- `nltk`

## Project Structure

```
Transformer_NMT/
├── src/                    # Core source code
│   ├── model.py           # Transformer model definition
│   ├── dataset.py         # Data processing and dataset classes
│   ├── utils.py           # Utility functions
│   └── visualize_training.py  # Training visualization
├── configs/                # Configuration files
│   ├── train.yaml         # Training configuration
│   └── inference.yaml     # Inference configuration
├── checkpoints/            # Model checkpoint directory
│   └── exp_*/             # Experiment-specific checkpoint directories
├── logs/                   # Training logs
├── outputs/                # Output results
├── train.py               # Training script
├── evaluation.py          # Evaluation script
└── inference.py           # Inference script
```

## Training, Evaluation, and Inference

### Train the Model

#### Single-GPU Training
```bash
python train.py
```

#### Multi-GPU Distributed Training
```bash
torchrun --nproc_per_node=<num_gpus> train.py
```

#### Configure Training Parameters
Edit the `configs/train.yaml` file to adjust training parameters, including:
- Model architecture parameters (`D_MODEL`, `NHEAD`, `NUM_ENCODER_LAYERS`, etc.)
- Training hyperparameters (`BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, etc.)
- Ablation experiment configurations (`POS_ENCODING_TYPE`, `NORM_TYPE`, etc.)

#### Ablation Experiment Configuration Examples

**Absolute Positional Encoding + LayerNorm**:
```yaml
POS_ENCODING_TYPE: absolute
NORM_TYPE: layernorm
```

**Relative Positional Encoding + RMSNorm**:
```yaml
POS_ENCODING_TYPE: relative
NORM_TYPE: rmsnorm
```

During training, model checkpoints will be automatically saved to the `checkpoints/exp_<experiment_name>/` directory, where the experiment name is generated automatically based on the configuration.

### Evaluate the Model

Evaluate the model's performance on the test set, outputting BLEU-1, BLEU-2, BLEU-3, BLEU-4, and Perplexity scores:

```bash
python evaluation.py model_path=<checkpoint_path>
```

Example:
```bash
python evaluation.py model_path=checkpoints/exp_abs_pos_ln_bs32_lre4/checkpoint_best.pth
```

### Inference for Translation

Use the trained model for single-sentence or batch translation:

```bash
python inference.py
```

Or specify the model path:
```bash
python inference.py model_path=<checkpoint_path>
```

## Configuration

### Training Configuration (`configs/train.yaml`)

Key configuration items:

- **Model Parameters**:
  - `D_MODEL`: Model dimension (default: 256)
  - `NHEAD`: Number of attention heads (default: 8)
  - `NUM_ENCODER_LAYERS`: Number of encoder layers (default: 4)
  - `NUM_DECODER_LAYERS`: Number of decoder layers (default: 4)
  - `DIM_FEEDFORWARD`: Feed-forward network dimension (default: 1024)
  - `DROPOUT`: Dropout rate (default: 0.1)
  - `MAX_LEN`: Maximum sequence length (default: 128)

- **Training Parameters**:
  - `BATCH_SIZE`: Batch size (default: 64)
  - `LEARNING_RATE`: Learning rate (default: 1e-5)
  - `NUM_EPOCHS`: Number of training epochs (default: 30)
  - `CLIP_GRAD`: Gradient clipping threshold (default: 5.0)
  - `LABEL_SMOOTHING`: Label smoothing coefficient (default: 0.1)

- **Ablation Experiment Parameters**:
  - `POS_ENCODING_TYPE`: Positional encoding type (`absolute` or `relative`)
  - `NORM_TYPE`: Normalization type (`layernorm` or `rmsnorm`)

## Features

- ✅ Complete Transformer architecture implementation
- ✅ Support for absolute and relative positional encoding
- ✅ Support for LayerNorm and RMSNorm
- ✅ Distributed training support (DDP)
- ✅ Mixed precision training (AMP)
- ✅ Automatic experiment management and checkpoint saving
- ✅ Comprehensive evaluation metrics (BLEU-1/2/3/4, Perplexity)
- ✅ Training process visualization
- ✅ Numerical stability optimization (NaN detection and handling)

## Acknowledgement

Thanks to the following repositories and projects:

- [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
- [Neural-Machine-Translation-Based-on-Transformer](https://github.com/piaoranyc/Neural-Machine-Translation-Based-on-Transformer)
- ⭐ [Transformer-NMT-Translation](https://github.com/Kwen-Chen/Transformer-NMT-Translation)
- [T5-base](https://huggingface.co/google-t5/t5-base/tree/main)
- [Text-to-Text Transfer Transformer](https://github.com/google-research/text-to-text-transfer-transformer)

## License

This project is for educational and research purposes only.
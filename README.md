# Vietnamese Visual Question Answering (VQA) - Seq2Seq

This repository implements a research-quality Vietnamese VQA system using CNN/ResNet image encoders, an LSTM question encoder, and an LSTM decoder with optional Bahdanau attention. The task is **sequence-to-sequence answer generation** (not classification).

## Repository Layout

- `config.yaml`: Global configuration
- `dataset/`: OpenViVQA JSON files
- `dataset/images/`: Image files referenced by `image_id`
- `embeddings/`: FastText vectors (`cc.vi.300.vec`)
- `data/`: Dataset and preprocessing utilities
- `models/`: Model components
- `training/`: Trainer, loss, metrics
- `utils/`: Tokenizer, vocabulary, beam search
- `checkpoints/`: Saved model weights
- `notebooks/train_vqa.ipynb`: End-to-end training notebook
- `demo/web_demo.py`: Gradio demo

## Dataset Placement

Place the OpenViVQA files as follows:

- `dataset/openvivqa_train_v2.json`
- `dataset/openvivqa_dev_v2.json`
- `dataset/openvivqa_test_v2.json`
- `dataset/images/` should contain all image files referenced in the JSON `images` table.

The dataset format must include:

- `images`: list of `{id, file_name}` objects
- `annotations`: list of `{image_id, question, answers}` objects

The training code uses the **first element** in the `answers` list as the target answer.

## FastText Embeddings

Download Vietnamese FastText vectors and place them at:

- `embeddings/cc.vi.300.vec`

Official download page:
- [FastText Vietnamese vectors](https://fasttext.cc/docs/en/crawl-vectors.html)

## Installation

```bash
pip install -r requirements.txt
```

## Training (Notebook)

Open and run the notebook:

```bash
jupyter notebook notebooks/train_vqa.ipynb
```

The notebook will:

1. Load config
2. Build vocabulary
3. Initialize models
4. Train and evaluate
5. Save checkpoints

## Training Notes

- **Teacher forcing** uses scheduled sampling with a decaying ratio.
- **Mixed precision** is enabled by default for Kaggle GPUs.
- **ResNet training** is performed in two stages:
  1. Freeze the backbone
  2. Fine-tune the last ResNet block

## Gradio Demo

After training, run:

```bash
python demo/web_demo.py
```

The UI outputs answers from four model variants side-by-side.

## Checkpoints

By default, checkpoints are saved to:

- `checkpoints/cnn_baseline.pth`
- `checkpoints/cnn_attention.pth`
- `checkpoints/resnet_baseline.pth`
- `checkpoints/resnet_attention.pth`

## Reproducibility

Set random seeds in the notebook using `training.trainer.set_seed` for reproducible experiments.

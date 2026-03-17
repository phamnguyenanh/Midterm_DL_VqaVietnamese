import os
import sys
from pathlib import Path
import yaml
import torch
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import build_vocab_from_json, load_fasttext_embeddings, load_dataset, parse_annotations
from data.dataloader import build_transforms
from data.dataset import VQADataset
from models.cnn_encoder import CNNEncoder
from models.resnet_encoder import ResNet50Encoder
from models.question_encoder import QuestionEncoder
from models.attention import BahdanauAttention
from models.fusion import FusionModule
from models.decoder import LSTMDecoder
from models.vqa_model import VQAModel
from training.loss import get_loss_fn


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return str(PROJECT_ROOT / path_value)


def resolve_config_paths(config: dict) -> dict:
    config = dict(config)
    config["data"] = dict(config["data"])
    config["data"]["train_json"] = resolve_path(config["data"]["train_json"])
    config["data"]["dev_json"] = resolve_path(config["data"]["dev_json"])
    config["data"]["test_json"] = resolve_path(config["data"]["test_json"])
    config["data"]["images_dir"] = resolve_path(config["data"]["images_dir"])
    config["data"]["embeddings_path"] = resolve_path(config["data"]["embeddings_path"])
    config["data"]["vocab_path"] = resolve_path(config["data"]["vocab_path"])
    return config


def check_dataset(config):
    data = load_dataset(config["data"]["train_json"])
    samples = parse_annotations(data)
    if not samples:
        print("No samples found in training dataset. Please provide OpenViVQA JSON files.")
        return False

    images_dir = config["data"]["images_dir"]
    image_id, _, _ = samples[0]
    id_map = {item.get("id"): item.get("file_name") or item.get("filename") for item in data.get("images", [])}
    filename = id_map.get(image_id)
    if filename is None:
        print("Image filename not found for the first sample. Check images metadata.")
        return False

    image_path = os.path.join(images_dir, filename)
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False

    return True


def build_vocab(config):
    vocab_path = config["data"]["vocab_path"]
    if os.path.exists(vocab_path):
        from utils.vocabulary import Vocabulary
        return Vocabulary.from_json(vocab_path)

    vocab = build_vocab_from_json(
        [
            config["data"]["train_json"],
            config["data"]["dev_json"],
            config["data"]["test_json"],
        ],
        max_size=config["data"]["vocab_size"],
    )
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    vocab.to_json(vocab_path)
    return vocab


def build_batch(config, vocab):
    transform = build_transforms()
    dataset = VQADataset(
        config["data"]["train_json"],
        config["data"]["images_dir"],
        vocab,
        max_question_len=config["data"]["max_question_len"],
        max_answer_len=config["data"]["max_answer_len"],
        transform=transform,
    )

    if len(dataset) == 0:
        print("Dataset is empty. Aborting smoke test.")
        sys.exit(0)

    batch_size = min(2, config["training"]["batch_size"], len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return next(iter(loader))


def build_model_variant(variant, config, vocab, embeddings):
    use_attention = "attention" in variant
    use_resnet = "resnet" in variant

    if use_resnet:
        image_encoder = ResNet50Encoder(pretrained=False)
        image_dim = config["model"]["resnet_out_dim"]
    else:
        image_encoder = CNNEncoder(out_dim=config["model"]["cnn_out_dim"])
        image_dim = config["model"]["cnn_out_dim"]

    question_encoder = QuestionEncoder(
        vocab_size=len(vocab),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_size=config["model"]["question_hidden_size"],
        padding_idx=vocab.pad_idx,
        embeddings=embeddings,
    )

    attention = None
    if use_attention:
        attention = BahdanauAttention(
            image_dim=image_dim,
            hidden_dim=config["model"]["decoder_hidden_size"],
            attn_dim=config["model"]["attention_dim"],
        )

    fusion = FusionModule(
        image_dim=image_dim,
        question_dim=config["model"]["question_hidden_size"],
        hidden_dim=config["model"]["decoder_hidden_size"],
    )

    decoder = LSTMDecoder(
        vocab_size=len(vocab),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["decoder_hidden_size"],
        image_dim=image_dim,
        padding_idx=vocab.pad_idx,
        attention=attention,
        embeddings=embeddings,
    )

    return VQAModel(image_encoder, question_encoder, fusion, decoder, use_attention=use_attention)


def main():
    config_path = os.environ.get("VQA_CONFIG", "config.yaml")
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    config = load_config(config_path)
    config = resolve_config_paths(config)

    if not check_dataset(config):
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = build_vocab(config)
    embeddings = load_fasttext_embeddings(
        config["data"]["embeddings_path"],
        vocab,
        embedding_dim=config["model"]["embedding_dim"],
    )

    images, questions, answers_input, answers_target = build_batch(config, vocab)
    images = images.to(device)
    questions = questions.to(device)
    answers_input = answers_input.to(device)
    answers_target = answers_target.to(device)

    variants = ["cnn_baseline", "cnn_attention", "resnet_baseline", "resnet_attention"]
    loss_fn = get_loss_fn(vocab.pad_idx)

    for variant in variants:
        model = build_model_variant(variant, config, vocab, embeddings).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(images, questions, answers_input, teacher_forcing_ratio=0.0)
            loss = loss_fn(logits.view(-1, logits.size(-1)), answers_target.view(-1))
            print(f"{variant} logits shape: {tuple(logits.shape)} | loss: {float(loss):.4f}")

            seqs = model.generate(
                images[:1],
                questions[:1],
                start_idx=vocab.start_idx,
                end_idx=vocab.end_idx,
                beam_width=3,
                max_len=config["data"]["max_answer_len"],
            )
            print(f"{variant} sample prediction indices: {seqs[0]}")

    model.train()
    optimizer = Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images, questions, answers_input, teacher_forcing_ratio=0.5)
    loss = loss_fn(logits.view(-1, logits.size(-1)), answers_target.view(-1))
    loss.backward()
    optimizer.step()
    print("Backward pass succeeded.")

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()


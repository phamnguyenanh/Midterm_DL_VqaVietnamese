import os
import sys
from pathlib import Path
import yaml
import torch
import gradio as gr
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import build_transforms
from data.preprocessing import normalize_text
from utils.tokenizer import tokenize
from utils.vocabulary import Vocabulary
from models.cnn_encoder import CNNEncoder
from models.resnet_encoder import ResNet50Encoder
from models.question_encoder import QuestionEncoder
from models.attention import BahdanauAttention
from models.fusion import FusionModule
from models.decoder import LSTMDecoder
from models.vqa_model import VQAModel


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(variant: str, vocab_size: int, pad_idx: int, embedding_dim: int, question_hidden: int, decoder_hidden: int):
    use_attention = "attention" in variant
    use_resnet = "resnet" in variant

    if use_resnet:
        image_encoder = ResNet50Encoder(pretrained=False)
        image_dim = 2048
    else:
        image_encoder = CNNEncoder(out_dim=256)
        image_dim = 256

    question_encoder = QuestionEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=question_hidden,
        padding_idx=pad_idx,
        embeddings=None,
    )

    attention = BahdanauAttention(image_dim=image_dim, hidden_dim=decoder_hidden, attn_dim=decoder_hidden) if use_attention else None
    fusion = FusionModule(image_dim=image_dim, question_dim=question_hidden, hidden_dim=decoder_hidden)

    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=decoder_hidden,
        image_dim=image_dim,
        padding_idx=pad_idx,
        attention=attention,
        embeddings=None,
    )

    return VQAModel(image_encoder, question_encoder, fusion, decoder, use_attention=use_attention)


def load_checkpoint(model, path: str, device: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    return model


def prepare_question(question: str, vocab: Vocabulary, max_len: int):
    tokens = tokenize(normalize_text(question))
    ids = vocab.encode(tokens)
    if len(ids) < max_len:
        ids = ids + [vocab.pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def predict_all(image: Image.Image, question: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = transform(image).unsqueeze(0).to(device)
    question_tensor = prepare_question(question, vocab, max_question_len).to(device)

    outputs = []
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            seq = model.generate(
                image_tensor,
                question_tensor,
                start_idx=vocab.start_idx,
                end_idx=vocab.end_idx,
                beam_width=beam_width,
                max_len=max_answer_len,
            )[0]
        tokens = vocab.decode(seq, stop_at_end=True)
        outputs.append(" ".join(tokens))
    return outputs


os.chdir(PROJECT_ROOT)
config = load_config(str(PROJECT_ROOT / "config.yaml"))
vocab = Vocabulary.from_json(config["data"]["vocab_path"])

transform = build_transforms()
max_question_len = config["data"]["max_question_len"]
max_answer_len = config["data"]["max_answer_len"]
beam_width = config["inference"]["beam_width"]

models = {}
variants = {
    "cnn_baseline": "cnn_baseline",
    "cnn_attention": "cnn_attention",
    "resnet_baseline": "resnet_baseline",
    "resnet_attention": "resnet_attention",
}

for key, variant in variants.items():
    m = build_model(
        variant=variant,
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        embedding_dim=config["model"]["embedding_dim"],
        question_hidden=config["model"]["question_hidden_size"],
        decoder_hidden=config["model"]["decoder_hidden_size"],
    )
    ckpt_path = config["checkpoints"][key]
    m = load_checkpoint(m, ckpt_path, device="cpu")
    m = m.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    models[key] = m


iface = gr.Interface(
    fn=predict_all,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Question", lines=2, placeholder="Nhập câu hỏi bằng tiếng Việt"),
    ],
    outputs=[
        gr.Textbox(label="Model 1: CNN Baseline"),
        gr.Textbox(label="Model 2: CNN + Attention"),
        gr.Textbox(label="Model 3: ResNet Baseline"),
        gr.Textbox(label="Model 4: ResNet + Attention"),
    ],
    title="Vietnamese VQA Demo",
    description="Upload an image and ask a Vietnamese question. The system generates answers from four model variants.",
)

if __name__ == "__main__":
    iface.launch()

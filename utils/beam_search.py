import torch
import torch.nn.functional as F


def beam_search_decode(
    decoder,
    image_global,
    image_regions,
    hidden,
    start_idx: int,
    end_idx: int,
    beam_width: int = 3,
    max_len: int = 15,
    device: str = "cpu",
):
    """Beam search decoding for a single example (batch size 1)."""
    if image_global.size(0) != 1:
        raise ValueError("Beam search expects batch size 1")

    sequences = [([start_idx], hidden, 0.0)]
    completed = []

    for _ in range(max_len):
        all_candidates = []
        for seq, h, score in sequences:
            if seq[-1] == end_idx:
                completed.append((seq, score))
                continue

            input_token = torch.tensor([seq[-1]], device=device)
            logits, h_new, _ = decoder.step(input_token, h, image_global, image_regions)
            log_probs = F.log_softmax(logits, dim=1).squeeze(0)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                next_token = int(topk_indices[k].item())
                next_score = score + float(topk_log_probs[k].item())
                all_candidates.append((seq + [next_token], (h_new[0].clone(), h_new[1].clone()), next_score))

        if not all_candidates:
            break

        all_candidates.sort(key=lambda x: x[2], reverse=True)
        sequences = all_candidates[:beam_width]

    if completed:
        completed.sort(key=lambda x: x[1], reverse=True)
        return completed[0][0]
    return sequences[0][0]

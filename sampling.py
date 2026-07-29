import torch


def sample_ids(model, ids, eot_token_id, max_tokens, ctx_len):
    for _ in range(max_tokens):
        # Crop current context if it exceeds the supported context size
        ids_cond = ids[:, -ctx_len:]

        with torch.no_grad():
            logits = model(ids_cond)

        logits = logits[:, -1, :]

        id_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        ids = torch.cat((ids, id_next), dim=1)  # (batch, n_tokens+1)

        if id_next.item() == eot_token_id:
            break

    return ids


def generate_text(model, encoding, eot_token_id, prompt, max_tokens, ctx_len, device):
    input_ids = torch.tensor(
        [eot_token_id] + encoding.encode_ordinary(prompt)
    ).unsqueeze(0)

    output_ids = sample_ids(
        model=model,
        ids=input_ids.to(device),
        eot_token_id=eot_token_id,
        max_tokens=max_tokens,
        ctx_len=ctx_len,
    )

    return encoding.decode(output_ids.squeeze(0).tolist())

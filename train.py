from dataclasses import asdict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from checkpoint import save_checkpoint
from config import TOKENIZER_VOCAB_SIZE, GPTConfig, TrainConfig
from data import load_tinystories, make_dataloader
from model import GPT
from sampling import generate_text
from tokenizer import build_encoding, train_tokenizer

CHECKPOINT_PATH = "checkpoints/model.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(train_config: TrainConfig | None = None) -> None:
    train_config = train_config or TrainConfig()
    torch.manual_seed(123)

    device = get_device()
    print(f"Using {device} device.")

    train_texts, val_texts = load_tinystories(
        train_config.num_train_stories, train_config.num_val_stories
    )

    tokenizer_state = train_tokenizer(
        train_texts + val_texts, vocab_size=TOKENIZER_VOCAB_SIZE
    )
    encoding = build_encoding(tokenizer_state)
    eot_token_id = tokenizer_state.eot_token_id

    train_ids = [eot_token_id]
    for document in train_texts:
        train_ids.extend(encoding.encode_ordinary(document) + [eot_token_id])

    val_ids = [eot_token_id]
    for document in val_texts:
        val_ids.extend(encoding.encode_ordinary(document) + [eot_token_id])

    gpt_config = GPTConfig(vocab_size=encoding.n_vocab)

    model = GPT(asdict(gpt_config))
    model.to(device)

    train_loader = make_dataloader(
        train_ids,
        batch_size=train_config.batch_size,
        max_length=gpt_config.ctx_len,
        stride=gpt_config.ctx_len,
        shuffle=True,
        drop_last=True,
    )
    val_loader = make_dataloader(
        val_ids,
        batch_size=train_config.batch_size,
        max_length=gpt_config.ctx_len,
        stride=gpt_config.ctx_len,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )

    train_losses, val_losses, tokens_seen = [], [], []
    step = 0

    val_prompt = train_texts[0][:25]

    epochs_bar = tqdm(total=train_config.num_epochs, desc="Epochs")
    batches_bar = tqdm(total=len(train_loader), desc="Batches", leave=False)
    val_bar = tqdm(total=len(val_loader), desc="Validation", leave=False)

    for epoch in range(train_config.num_epochs):
        batches_bar.reset()
        epoch_loss_sum = 0.0
        epoch_batch_count = 0

        for input_batch, target_batch in train_loader:
            model.train()
            optimizer.zero_grad()

            logits = model(input_batch.to(device))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_batch.to(device).reshape(-1),
            )
            loss.backward()
            optimizer.step()

            tokens_seen.append(input_batch.numel())
            step += 1

            batch_loss = loss.item()
            epoch_loss_sum += batch_loss
            epoch_batch_count += 1
            epoch_loss = epoch_loss_sum / epoch_batch_count

            batches_bar.set_postfix(loss=f"{batch_loss:.3f}")
            batches_bar.update(1)
            epochs_bar.set_postfix(loss=f"{epoch_loss:.3f}")

            if step % train_config.eval_freq == 0:
                model.eval()
                train_losses.append(epoch_loss)

                val_loss = 0.0
                val_bar.reset()
                with torch.no_grad():
                    for val_input, val_target in val_loader:
                        val_logits = model(val_input.to(device))
                        batch_val_loss = F.cross_entropy(
                            val_logits.reshape(-1, val_logits.size(-1)),
                            val_target.to(device).reshape(-1),
                        ).item()
                        val_loss += batch_val_loss
                        val_bar.set_postfix(loss=f"{batch_val_loss:.3f}")
                        val_bar.update(1)
                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                output_text = generate_text(
                    model=model,
                    encoding=encoding,
                    eot_token_id=eot_token_id,
                    prompt=val_prompt,
                    max_tokens=train_config.max_sample_tokens,
                    ctx_len=gpt_config.ctx_len,
                    device=device,
                )

                tqdm.write(
                    f"Epoch {epoch + 1} (Step {step:06d}): "
                    f"Train loss {epoch_loss:.3f}, Val loss {val_loss:.3f}"
                )
                tqdm.write(f"Output: {output_text!r}")

        epochs_bar.update(1)

    save_checkpoint(
        CHECKPOINT_PATH,
        model=model,
        gpt_config=gpt_config,
        tokenizer_state=tokenizer_state,
        train_losses=train_losses,
        val_losses=val_losses,
        tokens_seen=tokens_seen,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

from dataclasses import dataclass

import rustbpe
import tiktoken

from config import EOT_TOKEN


@dataclass
class TokenizerState:
    pat_str: str
    mergeable_ranks: dict[bytes, int]
    special_tokens: dict[str, int]
    eot_token_id: int


def train_tokenizer(corpus: list[str], vocab_size: int) -> TokenizerState:
    rustbpe_tokenizer = rustbpe.Tokenizer()
    rustbpe_tokenizer.train_from_iterator(corpus, vocab_size=vocab_size)

    mergeable_ranks = {
        bytes(key): rank for key, rank in rustbpe_tokenizer.get_mergeable_ranks()
    }
    eot_token_id = len(mergeable_ranks)

    return TokenizerState(
        pat_str=rustbpe_tokenizer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens={EOT_TOKEN: eot_token_id},
        eot_token_id=eot_token_id,
    )


def build_encoding(state: TokenizerState) -> tiktoken.Encoding:
    return tiktoken.Encoding(
        name="dataset_tokenizer",
        pat_str=state.pat_str,
        mergeable_ranks=state.mergeable_ranks,
        special_tokens=state.special_tokens,
    )

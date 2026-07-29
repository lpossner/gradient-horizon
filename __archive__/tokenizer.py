"""
BPE Tokenizer
"""

import os
import pickle
from functools import lru_cache
import rustbpe


SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer:
    """Light wrapper around tiktoken but train with rustbpe"""

    def __init__(self, encoder, bos_token):
        self.encoder = encoder
        self.bos_token_id = self.encode_special(bos_token)

    ## Factory methods
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # Set up rustbpe
        tokenizer = rustbpe.Tokenizer()
        # Special tokens are not trained here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        # Train tokenizer
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )
        # Construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        encoder = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,  # dict[str, int] (special token name -> token id)
        )
        return cls(encoder, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            encoder = pickle.load(f)
        return cls(encoder, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        encoder = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(encoder, "<|endoftext|>")

    ## Getter
    def get_vocab_size(self):
        return self.encoder.n_vocab

    def get_special_tokens(self):
        return self.encoder.special_tokens_set

    def get_bos_token_id(self):
        return self.bos_token_id

    ## Encode and decode
    def id_to_token(self, id):
        return self.encoder.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.encoder.encode_single_token(text)

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            ids = self.encoder.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)  # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.encoder.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)  # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.encoder.decode(ids)

    ## Helper
    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.encoder, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

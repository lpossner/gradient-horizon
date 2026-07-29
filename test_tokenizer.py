from tokenizer import TokenizerState, build_encoding, train_tokenizer

CORPUS = ["hello world", "hello there", "the quick brown fox"] * 5


def test_train_tokenizer_produces_working_encoding():
    state = train_tokenizer(CORPUS, vocab_size=300)
    encoding = build_encoding(state)

    ids = encoding.encode_ordinary("hello world")
    assert encoding.decode(ids) == "hello world"


def test_train_tokenizer_eot_token_id_matches_special_tokens():
    state = train_tokenizer(CORPUS, vocab_size=300)

    assert state.special_tokens == {"<|EOT|>": state.eot_token_id}
    assert state.eot_token_id == len(state.mergeable_ranks)


def test_build_encoding_from_reconstructed_state_matches_original():
    state = train_tokenizer(CORPUS, vocab_size=300)
    encoding = build_encoding(state)

    rebuilt_state = TokenizerState(
        pat_str=state.pat_str,
        mergeable_ranks=state.mergeable_ranks,
        special_tokens=state.special_tokens,
        eot_token_id=state.eot_token_id,
    )
    rebuilt_encoding = build_encoding(rebuilt_state)

    ids = encoding.encode_ordinary("the quick brown fox")
    assert rebuilt_encoding.decode(ids) == "the quick brown fox"
    assert rebuilt_encoding.n_vocab == encoding.n_vocab

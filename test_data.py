import data


class _FakeSplit:
    def __init__(self, texts):
        self._texts = texts

    def select(self, indices):
        return _FakeSplit([self._texts[i] for i in indices])

    def __getitem__(self, key):
        assert key == "text"
        return self._texts


def test_load_tinystories_slices_train_and_val(monkeypatch):
    fake_dataset = {
        "train": _FakeSplit([f"train story {i}" for i in range(10)]),
        "validation": _FakeSplit([f"val story {i}" for i in range(10)]),
    }
    monkeypatch.setattr(data, "load_dataset", lambda name: fake_dataset)

    train_texts, val_texts = data.load_tinystories(
        num_train_stories=3, num_val_stories=2
    )

    assert train_texts == ["train story 0", "train story 1", "train story 2"]
    assert val_texts == ["val story 0", "val story 1"]


def test_gpt_dataset_sliding_window():
    ids = list(range(10))
    dataset = data.GPTDataset(ids, max_length=4, stride=2)

    assert len(dataset) == 3
    input_ids, target_ids = dataset[0]
    assert input_ids.tolist() == [0, 1, 2, 3]
    assert target_ids.tolist() == [1, 2, 3, 4]

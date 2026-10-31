"""直接按 PDF 调用公开接口，不经 tests.adapters；完成 A/C 后运行。"""
from tokenizer_lab_code import Tokenizer, train_bpe
from tokenizer_lab_code.storage import save_vocab_merges


def test_pdf_train_save_load_encode_decode(tmp_path):
    source = tmp_path / "train.txt"
    source.write_text("hello hello 世界🙂<|endoftext|>hello world", encoding="utf-8")
    special = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path=str(source), vocab_size=280, special_tokens=special)
    assert isinstance(vocab, dict) and all(isinstance(k, int) and isinstance(v, bytes) for k, v in vocab.items())
    assert isinstance(merges, list) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in merges)
    assert 257 <= len(vocab) <= 280 and merges
    v, m = tmp_path / "vocab.json", tmp_path / "merges.json"
    save_vocab_merges(v, m, vocab, merges)
    a = Tokenizer(vocab=vocab, merges=merges, special_tokens=special)
    b = Tokenizer.from_files(vocab_filepath=str(v), merges_filepath=str(m), special_tokens=special)
    for text in ("", "hello 世界🙂", "new words", special[0], "hello"+special[0]+"世界"):
        ids = b.encode(text)
        assert ids == a.encode(text)
        assert b.decode(ids) == text
    stream = b.encode_iterable(iter(["hello\n", "世界🙂\n"]))
    assert iter(stream) is stream
    assert b.decode(list(stream)) == "hello\n世界🙂\n"

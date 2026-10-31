"""PDF 双文件接口的设施测试；不调用学生的 BPE/编码 TODO。"""
import json

import pytest

from tokenizer_lab_code import Tokenizer, train_bpe
from tokenizer_lab_code.cli import parser, run
from tokenizer_lab_code.common import TrainingResult
from tokenizer_lab_code.storage import load_model, save_model, save_vocab_merges


def test_from_files_preserves_bytes_ids_and_order(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"ab", 257: b"abc", 258: "你好".encode()})
    merges = [(b"a", b"b"), (b"ab", b"c")]
    v, m = tmp_path / "vocab.json", tmp_path / "merges.json"
    save_vocab_merges(v, m, vocab, merges)
    tokenizer = Tokenizer.from_files(vocab_filepath=str(v), merges_filepath=str(m), special_tokens=["<|endoftext|>"])
    assert tokenizer.merges == merges
    assert all(tokenizer.vocab[key] == raw for key, raw in vocab.items())
    assert tokenizer.vocab[259] == b"<|endoftext|>"
    assert len(vocab) == 259  # 不修改传入对象
    class Derived(Tokenizer):
        pass
    assert isinstance(Derived.from_files(v, m), Derived)


def test_double_file_validation_and_legacy_loader(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    v, m = tmp_path / "vocab.json", tmp_path / "merges.json"
    with pytest.raises(ValueError):
        save_vocab_merges(v, v, vocab, [])
    save_vocab_merges(v, m, vocab, [])
    with pytest.raises(ValueError):
        Tokenizer.from_files(m, v)
    content = json.loads(v.read_text())
    content["vocab"].append(content["vocab"][0])
    v.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="唯一"):
        Tokenizer.from_files(v, m)
    old = tmp_path / "tokenizer.json"
    save_model(old, TrainingResult(vocab, []), special_tokens=[])
    assert load_model(old).vocab == vocab


def test_public_train_and_cli_double_export_without_merges(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("ab ab", encoding="utf-8")
    vocab, merges = train_bpe(input_path=str(source), vocab_size=256, special_tokens=[])
    assert len(vocab) == 256 and merges == []
    output, v, m = [tmp_path / name for name in ("model.json", "vocab.json", "merges.json")]
    args = ["train", "--input", str(source), "--vocab-size", "256", "--no-special-tokens",
            "--output", str(output), "--vocab-output", str(v), "--merges-output", str(m)]
    run(parser().parse_args(args))
    assert Tokenizer.from_files(v, m).vocab == load_model(output).vocab
    with pytest.raises(ValueError, match="同时"):
        run(parser().parse_args(args[:-2]))
    args[-1] = str(output)
    with pytest.raises(ValueError, match="不同"):
        run(parser().parse_args(args))

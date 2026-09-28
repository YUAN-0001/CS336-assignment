"""教师提供：明确使用十六进制保存 bytes；缓存和词表不能互相混用。"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import TrainingResult, validate_special_tokens
from .pretokenize import PATTERN
from .tokenizer import Tokenizer


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_cache(path, counts: Counter[bytes], *, special_tokens, pattern=PATTERN, source=None) -> None:
    write_json(
        path,
        {
            "format": "cs336-pretoken-counts-v1",
            "special_tokens": validate_special_tokens(special_tokens),
            "pattern": pattern,
            "source": str(source) if source is not None else None,
            "source_sha256": file_sha256(source) if source is not None else None,
            "counts": [[raw.hex(), count] for raw, count in sorted(counts.items())],
        },
    )


def load_cache(path) -> tuple[Counter[bytes], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "cs336-pretoken-counts-v1":
        raise ValueError("不是本实验的 pre-token 缓存")
    validate_special_tokens(data["special_tokens"])
    if not isinstance(data["pattern"], str) or not data["pattern"]:
        raise ValueError("缓存缺少有效 pattern")
    counts = Counter()
    for raw, count in data["counts"]:
        token = bytes.fromhex(raw)
        if not token or token in counts or type(count) is not int or count <= 0:
            raise ValueError("缓存 token 必须非空、唯一，次数必须为正整数")
        counts[token] = count
    return counts, data


def save_model(path, result: TrainingResult, *, special_tokens, pattern=PATTERN, metadata=None) -> None:
    # 使用构造器检查格式，但不调用待完成的 C1/C2。
    Tokenizer(result.vocab, result.merges, special_tokens, pattern=pattern)
    write_json(
        path,
        {
            "format": "cs336-bpe-model-v1",
            "special_tokens": special_tokens,
            "pattern": pattern,
            "vocab": [[i, raw.hex()] for i, raw in sorted(result.vocab.items())],
            "merges": [[a.hex(), b.hex()] for a, b in result.merges],
            "trace": [
                {"step": s.step, "pair": [x.hex() for x in s.pair], "frequency": s.frequency} for s in result.trace
            ],
            "metadata": metadata or {},
        },
    )


def load_model(path) -> Tokenizer:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "cs336-bpe-model-v1":
        raise ValueError("不是本实验的 BPE 词表文件")
    vocab = {i: bytes.fromhex(raw) for i, raw in data["vocab"]}
    if len(vocab) != len(data["vocab"]):
        raise ValueError("词表文件包含重复 ID")
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]
    return Tokenizer(vocab, merges, data["special_tokens"], pattern=data["pattern"])


def save_vocab_merges(vocab_filepath, merges_filepath, vocab, merges):
    """PDF 双文件接口。bytes 用 hex 无损表示；特殊 token 列表由构造器单独接收。

    本格式用于 PDF 的默认预分词规则。自定义 pattern 的旧合并文件仍用
    save_model/load_model 保存完整配置，不能误以为双文件包含 pattern。
    """
    if Path(vocab_filepath).resolve() == Path(merges_filepath).resolve():
        raise ValueError("vocab 和 merges 必须保存到两个不同文件")
    checked = Tokenizer(vocab, merges)
    write_json(vocab_filepath, {
        "format": "cs336-vocab-hex-v1",
        "vocab": [[i, raw.hex()] for i, raw in sorted(checked.vocab.items())],
    })
    write_json(merges_filepath, {
        "format": "cs336-merges-hex-v1",
        "merges": [[a.hex(), b.hex()] for a, b in checked.merges],
    })


def load_vocab_merges(vocab_filepath, merges_filepath):
    """读取 save_vocab_merges 的双文件；格式校验不调用任何学生 TODO。"""
    vocab_data = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
    merges_data = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
    if not isinstance(vocab_data, dict) or vocab_data.get("format") != "cs336-vocab-hex-v1":
        raise ValueError("词表文件格式错误；应由 save_vocab_merges 保存")
    if not isinstance(merges_data, dict) or merges_data.get("format") != "cs336-merges-hex-v1":
        raise ValueError("合并表文件格式错误；不能传入旧的 tokenizer.json 或缓存")
    vocab = {}
    for token_id, hex_bytes in vocab_data["vocab"]:
        if type(token_id) is not int or token_id in vocab:
            raise ValueError("词表 ID 必须是唯一整数")
        vocab[token_id] = bytes.fromhex(hex_bytes)
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_data["merges"]]
    checked = Tokenizer(vocab, merges)
    return checked.vocab, checked.merges

"""这些测试不调用任何学生 TODO；用于确认框架和环境可用。"""

from dataclasses import asdict
import io
import json

import numpy as np
import pytest
import torch

from lab_tools.check import targets
from lab_tools.configuration import TrainConfig, load_config
from lab_tools.plot import plot
from lab_tools.submission import validate_submission
from training_lab_code.checkpoint import load_checkpoint, save_checkpoint
from transformer_lab_code.attention import apply_mask, merge_heads, split_heads
from transformer_lab_code.model import ModelConfig, TransformerLM


def test_model_parameters_registered():
    config = ModelConfig(vocab_size=288, context_length=32, d_model=64, num_layers=2, num_heads=4, d_ff=192)
    model = TransformerLM(**asdict(config))
    expected = 2 * 288 * 64 + 2 * (4 * 64**2 + 3 * 64 * 192 + 2 * 64) + 64
    assert sum(p.numel() for p in model.parameters()) == expected
    assert "layers.0.attn.q_proj.weight" in model.state_dict()
    assert not any("rope" in name for name in model.state_dict())
    assert len(list(model.named_buffers())) == 4


def test_heads_roundtrip_and_mask():
    x = torch.arange(2 * 3 * 16).reshape(2, 3, 16)
    assert split_heads(x, 4).shape == (2, 4, 3, 4)
    torch.testing.assert_close(merge_heads(split_heads(x, 4)), x)
    mask = torch.tensor([[True, False]])
    scores = apply_mask(torch.zeros(2, 1, 2), mask)
    assert (scores[..., 0] == 0).all() and torch.isneginf(scores[..., 1]).all()


def test_checkpoint_filelike_and_rng():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stream = io.BytesIO()
    save_checkpoint(model, optimizer, 7, stream)
    expected_torch, expected_numpy = torch.rand(3), np.random.rand(3)
    stream.seek(0)
    restored = torch.nn.Linear(2, 2)
    assert load_checkpoint(stream, restored, optimizer, restore_random=True) == 7
    torch.testing.assert_close(torch.rand(3), expected_torch)
    np.testing.assert_array_equal(np.random.rand(3), expected_numpy)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_stage_boundaries():
    assert all("tokenizer" in name for name in targets(1))
    assert "tests/test_optimizer.py" not in targets(2)
    assert "tests/test_data.py" not in targets(3)
    assert "milestone_tests/test_m5.py" not in targets(4)
    assert set(targets(4)) < set(targets(5))


def test_configuration_rejects_typo_and_invalid_dims(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"vocab_size": 0}, "training": {"batc_size": 4}}))
    with pytest.raises(TypeError):
        load_config(path, 288)
    with pytest.raises(ValueError):
        ModelConfig(d_model=65)
    with pytest.raises(ValueError):
        TrainConfig(gradient_accumulation=0)
    assert TrainConfig().cpu_threads is None


def test_svg_plot(tmp_path):
    run = tmp_path / "a"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(dict(step=i, tokens=i * 8, wall_seconds=i, val_loss=3 - i)) for i in range(2))
    )
    plot([run], tmp_path / "curve.svg")
    assert "polyline" in (tmp_path / "curve.svg").read_text()


@pytest.mark.parametrize("stage", [4, 5])
def test_submission_without_members(tmp_path, stage):
    """提交格式只依赖实验产物；以下为验证器的最小合成测试夹具。"""
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    runs = []
    for role in (["baseline"] if stage == 4 else ["baseline", "candidate"]):
        directory = tmp_path / role
        directory.mkdir()
        write(directory / "summary.json", dict(status="complete", step=1, tokens=4,
                                               val_step=1, val_loss=2.0, val_tokens=4))
        write(directory / "config.json", dict(model=dict(context_length=4), training=dict(
            total_steps=1, batch_size=1, gradient_accumulation=1,
            eval_context_length=4, eval_batch_size=1, eval_batches=1)))
        write(directory / "provenance.json", dict(data_identity="test-fixture"))
        (directory / "metrics.jsonl").write_text("\n".join(json.dumps(dict(
            step=i, tokens=4*i, wall_seconds=i, val_loss=3.0-i)) for i in (0, 1)))
        torch.save({}, directory / "last.pt")
        runs.append(dict(role=role, path=role))
    (tmp_path / "report.md").write_text("验证器测试夹具。" * 20, encoding="utf-8")
    plot([tmp_path / record["path"] for record in runs], tmp_path / "curve.svg")
    write(tmp_path / "sample.json", dict(data_identity="test-fixture", prompt_ids=[1],
          continuation="a", continuation_ids=[2], max_new_tokens=1))
    write(tmp_path / "verification.json", dict(overfit_initial_loss=4.0,
          overfit_final_loss=1.0, recovery_max_abs_diff=0.0))
    submission = dict(milestone=stage, report="report.md", curve="curve.svg", runs=runs,
                      samples=["sample.json"], verification="verification.json",
                      comparison="tokens", hypothesis="这是用于验证提交格式的测试假设")
    path = tmp_path / "submission.json"
    write(path, submission)
    assert validate_submission(path, stage) == submission
    submission["report"] = "missing.md"
    write(path, submission)
    with pytest.raises(ValueError, match="缺少 report"):
        validate_submission(path, stage)

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.nn.functional as F

from sequence_policy.checkpointing import save_checkpoint, load_checkpoint
from sequence_policy.model import CausalMoveTransformer

VOCAB = 20483
KW = dict(vocab_size=VOCAB, d_model=64, num_layers=2, num_heads=4, head_dim=16,
          ffn_hidden=128, context_length=32, dropout=0.0, attention_dropout=0.0)


def make_model_and_opt():
    torch.manual_seed(0)
    m = CausalMoveTransformer(**KW)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, opt


# 18+19. Guardado/carga de state_dict + checkpoint completo
def test_checkpoint_save_and_load_full_state(tmp_path):
    m, opt = make_model_and_opt()
    x = torch.randint(0, VOCAB, (2, 8))
    labels = torch.randint(0, VOCAB, (2, 8))
    logits, _ = m(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
    loss.backward()
    opt.step()

    ckpt_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(ckpt_path, m, optimizer=opt, global_step=42, tokens_seen=1234,
                     config={"a": 1}, dataset_hashes={"h": "x"}, split_hashes={"s": "y"})
    assert os.path.isfile(ckpt_path)

    m2, opt2 = make_model_and_opt()
    ckpt = load_checkpoint(ckpt_path, m2, optimizer=opt2,
                            expected_config={"a": 1}, expected_dataset_hashes={"h": "x"},
                            expected_split_hashes={"s": "y"})
    assert ckpt["global_step"] == 42
    assert ckpt["tokens_seen"] == 1234
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


# 20. Reanudacion: continuidad exacta de global_step/tokens_seen
def test_resume_continues_exact_counters(tmp_path):
    m, opt = make_model_and_opt()
    ckpt_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(ckpt_path, m, optimizer=opt, global_step=100, tokens_seen=5000, epoch_equivalent=0.5)

    m2, opt2 = make_model_and_opt()
    ckpt = load_checkpoint(ckpt_path, m2, optimizer=opt2)
    global_step = ckpt["global_step"]
    tokens_seen = ckpt["tokens_seen"]
    assert global_step == 100
    assert tokens_seen == 5000
    # simular continuar entrenando desde ahi
    global_step += 1
    assert global_step == 101


def test_checkpoint_incompatible_config_refuses_load(tmp_path):
    m, opt = make_model_and_opt()
    ckpt_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(ckpt_path, m, optimizer=opt, config={"context_length": 128})

    m2, opt2 = make_model_and_opt()
    try:
        load_checkpoint(ckpt_path, m2, optimizer=opt2, expected_config={"context_length": 256})
        assert False, "deberia haber rechazado el checkpoint incompatible"
    except ValueError:
        pass


# 26. Exportacion y recarga
def test_export_and_reload_state_dict(tmp_path):
    m, _ = make_model_and_opt()
    m.eval()
    x = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        logits_before, _ = m(x)

    export_path = str(tmp_path / "exported.pt")
    torch.save({"model_state": m.state_dict(), "config": KW}, export_path)

    reloaded = CausalMoveTransformer(**KW)
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(payload["model_state"])
    reloaded.eval()
    with torch.no_grad():
        logits_after, _ = reloaded(x)
    assert torch.equal(logits_before, logits_after)

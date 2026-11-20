"""Pruebas de integracion end-to-end de scripts/run_full_train.py, usando
--local-structural-test (arquitectura TINY, permitida en CPU) contra el
checkpoint smoke.pt existente como "piloto" simulado. Corre el script real
via subprocess -- exactamente el mismo camino de codigo que se usaria en
Colab, solo que con TINY en vez de BASE.

Marca STRUCTURALLY_VALIDATED_LOCAL_CPU. La validacion de AMP/OOM reales en
GPU queda GPU_RUNTIME_VALIDATION_PENDING_COLAB."""
import glob
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_full_train.py")
PILOT_TINY_CKPT = os.path.join(REPO_ROOT, "checkpoints", "smoke_tiny", "smoke.pt")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(PILOT_TINY_CKPT),
    reason="requiere checkpoints/smoke_tiny/smoke.pt (Etapa D) como 'piloto' simulado para el smoke",
)


def run_script(output_dir, max_steps, confirm=True, extra_args=None, timeout=300):
    args = [sys.executable, SCRIPT, "--local-structural-test",
            "--pilot-checkpoint", PILOT_TINY_CKPT, "--output-dir", output_dir,
            "--max-steps", str(max_steps)]
    if confirm:
        args.append("--confirm-full-training")
    if extra_args:
        args += extra_args
    result = subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT, timeout=timeout)
    return result


def test_no_confirm_flag_does_not_train(tmp_path):
    out_dir = str(tmp_path / "run_no_confirm")
    result = run_script(out_dir, max_steps=5, confirm=False)
    assert result.returncode == 0
    assert not os.path.isfile(os.path.join(out_dir, "last.ckpt")), \
        "sin --confirm-full-training NO debe escribirse ningun checkpoint"


def test_fresh_run_warm_starts_with_new_optimizer_scheduler(tmp_path):
    """El primer checkpoint debe declarar warm_start_from_pilot (no resume),
    y run_global_step debe arrancar en 0, no heredar el del piloto (7532
    para el piloto real BASE, o el que tenga smoke.pt)."""
    out_dir = str(tmp_path / "run_fresh")
    result = run_script(out_dir, max_steps=3)
    assert result.returncode == 0, result.stderr[-2000:]

    events_path = os.path.join(out_dir, "resume_log_events.jsonl")
    events = [json.loads(line) for line in open(events_path, encoding="utf-8")]
    assert events[0]["event"] == "warm_start_from_pilot"

    ckpt_audit = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "audit_full_checkpoint.py"),
         "--checkpoint", os.path.join(out_dir, "last.ckpt")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    audit_result = json.loads(ckpt_audit.stdout.split("\n\n")[0]) if "\n\n" in ckpt_audit.stdout \
        else json.loads(ckpt_audit.stdout[:ckpt_audit.stdout.rfind("]") + 1])
    assert audit_result[0]["ok"] is True
    assert audit_result[0]["run_global_step"] == 3  # NO 7532+algo -- run_global_step es propio de esta corrida


def test_resume_continues_exact_step_and_lr(tmp_path):
    out_dir = str(tmp_path / "run_resume")
    r1 = run_script(out_dir, max_steps=3)
    assert r1.returncode == 0, r1.stderr[-2000:]

    audit1 = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "audit_full_checkpoint.py"),
         "--checkpoint", os.path.join(out_dir, "last.ckpt")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    a1 = json.loads(audit1.stdout[:audit1.stdout.rfind("]") + 1])[0]
    lr_after_first_run = a1["current_lr"]

    r2 = run_script(out_dir, max_steps=6)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert "resumed_from_last_ckpt" in r2.stdout

    audit2 = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "audit_full_checkpoint.py"),
         "--checkpoint", os.path.join(out_dir, "last.ckpt")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    a2 = json.loads(audit2.stdout[:audit2.stdout.rfind("]") + 1])[0]
    assert a2["run_global_step"] == 6
    # LR debe haber avanzado (scheduler restaurado, no reiniciado a step 0)
    assert a2["current_lr"] != lr_after_first_run


def test_corrupted_last_ckpt_falls_back_to_periodic(tmp_path):
    out_dir = str(tmp_path / "run_corrupt")
    r1 = run_script(out_dir, max_steps=3)
    assert r1.returncode == 0, r1.stderr[-2000:]

    last_ckpt = os.path.join(out_dir, "last.ckpt")
    with open(last_ckpt, "wb") as f:
        f.write(b"esto no es un checkpoint valido")

    r2 = run_script(out_dir, max_steps=5)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert "last_ckpt_corrupted_or_incompatible" in r2.stdout
    assert "recovered_from_periodic_checkpoint" in r2.stdout


def test_no_test_split_ever_referenced_in_script_source():
    """Escaneo estatico: el script no debe mencionar test_games ni
    windows_test en ningun punto -- deteccion explicita, no solo confianza."""
    src = open(SCRIPT, encoding="utf-8").read()
    assert "test_games" not in src
    assert "windows_test" not in src
    assert "'test'" not in src and '"test"' not in src.replace('"tested"', "")


def test_run_manifest_written_with_expected_fields(tmp_path):
    out_dir = str(tmp_path / "run_manifest_test")
    result = run_script(out_dir, max_steps=2)
    assert result.returncode == 0, result.stderr[-2000:]
    manifest = json.load(open(os.path.join(out_dir, "run_manifest.json"), encoding="utf-8"))
    assert manifest["classification"] == "WEIGHTS_WARM_START_FROM_PILOT_FULL_NEW_EPOCH"
    assert manifest["run_global_step"] == 2
    assert "cumulative_tokens_seen_including_pilot" in manifest
    assert manifest["local_structural_test"] is True


def test_capped_max_steps_is_idempotent_across_reinvocations(tmp_path):
    """Con el mismo tope --max-steps en dos invocaciones sucesivas, la
    segunda debe resumir y detenerse de inmediato sin agregar pasos extra
    (run_global_step no avanza mas alla del tope pedido). El caso real de
    'total_steps del scheduler ya alcanzado' se audita por separado en
    FULL_TRAINING_PIPELINE_BUILD.md (requiere ~18k pasos para TINY en CPU,
    no practico de reproducir en un test unitario rapido)."""
    out_dir = str(tmp_path / "run_finished")
    r1 = run_script(out_dir, max_steps=2)
    assert r1.returncode == 0
    audit1 = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "audit_full_checkpoint.py"),
         "--checkpoint", os.path.join(out_dir, "last.ckpt")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    step_after_r1 = json.loads(audit1.stdout[:audit1.stdout.rfind("]") + 1])[0]["run_global_step"]
    assert step_after_r1 == 2

    r2 = run_script(out_dir, max_steps=2)  # mismo tope: no deberia agregar pasos
    assert r2.returncode == 0
    audit2 = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "audit_full_checkpoint.py"),
         "--checkpoint", os.path.join(out_dir, "last.ckpt")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    step_after_r2 = json.loads(audit2.stdout[:audit2.stdout.rfind("]") + 1])[0]["run_global_step"]
    assert step_after_r2 == 2, "no debe reentrenar mas alla del tope ya alcanzado"


def test_micro_steps_per_optimizer_step_match_grad_accum_target(tmp_path):
    """Prueba de regresion DIRECTA del bug real reportado: antes
    optimizer_step avanzaba 1:1 con cada microbatch. Ahora debe hacer falta
    grad_accum_target microbatches por cada optimizer step."""
    out_dir = str(tmp_path / "run_accum_check")
    result = run_script(out_dir, max_steps=3)
    assert result.returncode == 0, result.stderr[-2000:]
    manifest = json.load(open(os.path.join(out_dir, "run_manifest.json"), encoding="utf-8"))
    acc = manifest["accumulation_accounting_this_invocation"]
    assert acc["optimizer_steps"] == 3
    assert acc["micro_steps"] == 3 * acc["grad_accum_target"]
    assert acc["real_grad_accum_observed"] == acc["grad_accum_target"]
    assert manifest["run_global_step"] == 3, "run_global_step debe ser optimizer steps, no microbatches"


def test_partial_group_at_end_of_epoch_applied_not_discarded(tmp_path):
    """Dataset deliberadamente chico (50 ventanas, microbatch=4 -> 13 batches
    totales) para forzar el fin real de la epoca a mitad de un grupo de
    acumulacion (grad_accum_target=12 en CPU_LOCAL: 12+1). El grupo parcial
    de 1 microbatch debe aplicarse (optimizer_step=2), no descartarse."""
    out_dir = str(tmp_path / "run_partial_group")
    result = run_script(out_dir, max_steps=100, extra_args=[
        "--local-structural-test-max-windows", "50", "--dev-fast-max-windows", "20",
    ])
    assert result.returncode == 0, result.stderr[-2000:]
    assert "grupo parcial de fin de epoca" in result.stdout
    manifest = json.load(open(os.path.join(out_dir, "run_manifest.json"), encoding="utf-8"))
    acc = manifest["accumulation_accounting_this_invocation"]
    # 13 microbatches totales: 1 grupo completo (12) + 1 grupo parcial (1) = 2 optimizer steps
    assert acc["optimizer_steps"] == 2
    assert acc["micro_steps"] == 13
    assert manifest["run_global_step"] == 2


def test_resume_at_optimizer_step_boundary_preserves_accumulation_accounting(tmp_path):
    """Reanudar despues de un run con grupos completos debe seguir contando
    optimizer steps correctamente en la segunda invocacion (no reinicia la
    nocion de grad_accum_target ni cuenta microbatches como si fueran steps)."""
    out_dir = str(tmp_path / "run_resume_accum")
    r1 = run_script(out_dir, max_steps=2)
    assert r1.returncode == 0, r1.stderr[-2000:]
    m1 = json.load(open(os.path.join(out_dir, "run_manifest.json"), encoding="utf-8"))
    assert m1["run_global_step"] == 2

    r2 = run_script(out_dir, max_steps=5)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert "resumed_from_last_ckpt" in r2.stdout
    m2 = json.load(open(os.path.join(out_dir, "run_manifest.json"), encoding="utf-8"))
    assert m2["run_global_step"] == 5
    acc2 = m2["accumulation_accounting_this_invocation"]
    # esta invocacion SOLO cubre los pasos 3,4,5 (3 optimizer steps nuevos)
    assert acc2["optimizer_steps"] == 3
    assert acc2["micro_steps"] == 3 * acc2["grad_accum_target"]


def test_best_checkpoints_created_correctly_under_real_accumulation(tmp_path):
    out_dir = str(tmp_path / "run_best_ckpt_accum")
    result = run_script(out_dir, max_steps=2, extra_args=["--dev-fast-max-windows", "30"])
    assert result.returncode == 0, result.stderr[-2000:]
    assert os.path.isfile(os.path.join(out_dir, "best_mrr.ckpt"))
    assert os.path.isfile(os.path.join(out_dir, "best_top10.ckpt"))
    audit = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "audit_full_checkpoint.py"),
         "--checkpoint", os.path.join(out_dir, "best_mrr.ckpt")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    a = json.loads(audit.stdout[:audit.stdout.rfind("]") + 1])[0]
    assert a["ok"] is True
    assert a["dev_metrics_recent"]["mrr"] is not None
    # el checkpoint best_mrr debe corresponder a un limite seguro (run_global_step
    # es un conteo de optimizer steps entero, nunca fraccionario a mitad de grupo)
    assert isinstance(a["run_global_step"], int) and a["run_global_step"] >= 0


def test_checkpoint_accumulation_metadata_always_at_safe_boundary(tmp_path):
    """El schema del checkpoint debe reflejar micro_step_at_save == 0 siempre
    (invariante de diseno: nunca se persisten gradientes a mitad de grupo)."""
    out_dir = str(tmp_path / "run_accum_metadata")
    result = run_script(out_dir, max_steps=2)
    assert result.returncode == 0, result.stderr[-2000:]
    import torch
    ckpt = torch.load(os.path.join(out_dir, "last.ckpt"), map_location="cpu", weights_only=False)
    accum = ckpt["extra"]["run_config"]["accumulation"]
    assert accum["micro_step_at_save"] == 0
    assert accum["grad_accum_target"] > 1

"""Padding dinamico por batch (bucketing, seccion 7/11 del pedido)."""
import torch

from sequence_policy.constants import PAD_ID
from sequence_policy.dataset_reader import SequenceWindowDataset


class SequenceBatchCollator:
    def __init__(self, pad_id: int = PAD_ID, ignore_index: int = SequenceWindowDataset.IGNORE_INDEX):
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    def __call__(self, batch: list) -> dict:
        max_len = max(item["length"] for item in batch)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            n = item["length"]
            input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            labels[i, :n] = torch.tensor(item["labels"], dtype=torch.long)
            attention_mask[i, :n] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        }


class ExtendedBatchCollator(SequenceBatchCollator):
    """Igual que SequenceBatchCollator, mas metadata por ventana (context_start,
    sample_id) necesaria para segmentar por ply/longitud de contexto en la
    evaluacion ampliada. No se usa en entrenamiento ni en run_eval.py original."""

    def __call__(self, batch: list) -> dict:
        out = super().__call__(batch)
        out["_context_start"] = torch.tensor([item["context_start"] for item in batch], dtype=torch.long)
        out["_sample_id"] = [item["sample_id"] for item in batch]
        return out

"""DeterministicWindowSampler: orden reproducible, bucketing por longitud
(para no desperdiciar padding) y reanudacion exacta (seccion 8 del pedido).

Estrategia: permutacion global determinista (seed + epoch_equivalent) de
todos los indices de ventana, agrupada en mega-lotes de tamano
`batch_size * bucket_mult` que se ordenan internamente por longitud antes de
cortarlos en batches reales -- conserva la mayor parte de la aleatoriedad
mientras agrupa longitudes parecidas en el mismo batch."""
import numpy as np


class DeterministicWindowSampler:
    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int,
                 bucket_mult: int = 50, epoch_equivalent: int = 0):
        self.lengths = np.asarray(lengths)
        self.n = len(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_mult = bucket_mult
        self.epoch_equivalent = epoch_equivalent

        self._batches = None
        self._batch_pos = 0
        self._samples_consumed = 0
        self._tokens_consumed = 0
        self._build_batches()

    def _build_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch_equivalent)
        perm = rng.permutation(self.n)
        mega = self.batch_size * self.bucket_mult
        batches = []
        for i in range(0, self.n, mega):
            chunk = perm[i:i + mega]
            chunk_sorted = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for j in range(0, len(chunk_sorted), self.batch_size):
                batches.append(chunk_sorted[j:j + self.batch_size])
        # Barajar el ORDEN de los batches (no su contenido) para no entrenar
        # sistematicamente cortas->largas dentro de cada mega-lote.
        batch_order = rng.permutation(len(batches))
        self._batches = [batches[i] for i in batch_order]

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        for i in range(self._batch_pos, len(self._batches)):
            batch_idx = self._batches[i]
            self._batch_pos = i + 1
            self._samples_consumed += len(batch_idx)
            self._tokens_consumed += int(self.lengths[batch_idx].sum())
            yield batch_idx.tolist()

    def state_dict(self) -> dict:
        return {
            "seed": self.seed, "epoch_equivalent": self.epoch_equivalent,
            "batch_size": self.batch_size, "bucket_mult": self.bucket_mult,
            "batch_pos": self._batch_pos, "samples_consumed": self._samples_consumed,
            "tokens_consumed": self._tokens_consumed, "n_batches_total": len(self._batches),
        }

    def load_state_dict(self, state: dict):
        assert state["seed"] == self.seed
        assert state["epoch_equivalent"] == self.epoch_equivalent
        assert state["batch_size"] == self.batch_size
        assert state["bucket_mult"] == self.bucket_mult
        self._batch_pos = state["batch_pos"]
        self._samples_consumed = state["samples_consumed"]
        self._tokens_consumed = state["tokens_consumed"]

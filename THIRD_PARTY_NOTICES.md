# Third-party notices

Chess Alpha's own source code is licensed under the **GNU General Public License v3.0 or later** (see [`LICENSE`](LICENSE)). That license covers only the code written for this project. It grants no rights over the third-party software, data or engines listed below, each of which keeps its own license.

**This repository bundles no third-party code, binaries, data or model weights.** Every dependency below is installed separately by the user, for example with `pip install -r requirements.txt`.

## Python dependencies (installed separately, not redistributed)

| Package | Used for | License (from the installed package metadata) | Compatible with GPL-3.0-or-later |
|---|---|---|---|
| [python-chess](https://github.com/niklasf/python-chess) (`chess`) | board representation, move generation, PGN/FEN/UCI, Polyglot Zobrist hashing | GPL-3.0-or-later | Yes. It is the reason this project uses the GPL |
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause (plus bundled 0BSD, MIT, Zlib, CC0-1.0 parts) | Yes |
| [pandas](https://pandas.pydata.org) | tabular data | BSD-3-Clause | Yes |
| [PyArrow](https://arrow.apache.org) | Parquet I/O | Apache-2.0 | Yes (Apache-2.0 is compatible with GPLv3) |
| [PyYAML](https://pyyaml.org) | configuration files | MIT | Yes |
| [psutil](https://github.com/giampaolo/psutil) | resource monitoring | BSD-3-Clause | Yes |
| [pytest](https://pytest.org) | tests only | MIT | Yes |
| [PyTorch](https://pytorch.org) (optional) | neural models | BSD-3-Clause | Yes |
| [DuckDB](https://duckdb.org) (optional) | analytical queries | MIT | Yes |
| [tqdm](https://github.com/tqdm/tqdm) (optional) | progress bars | MPL-2.0 AND MIT | Yes (MPL-2.0 allows combination with GPL code) |

The licenses above were read from the metadata of the versions validated for this release (see `requirements.txt` and `requirements-optional.txt`). Check the upstream projects for the terms of other versions.

## External programs (never included)

- **Stockfish** (GPL-3.0). Some optional scripts call a Stockfish binary that you install yourself, as a separate process through the UCI protocol (via python-chess). No Stockfish source code or binary is in this repository. Point the `STOCKFISH_PATH` environment variable at your own binary to use those scripts. The tests and examples do not need it.
- **Syzygy endgame tablebases**. `tokenization_enrichment_v3/scripts/tablebase.py` can read local tablebase files if `SYZYGY_PATH` is set. It never downloads anything, and no tablebase files are included.

## Data and model weights (not distributed)

- **Game data.** The PGN collections used for training and evaluation are **not** included. Their provenance and redistribution terms have not been confirmed, so no raw or derived dataset is published.
- **Model checkpoints.** The trained Epoch4 and Sequence Policy weights are **not** published. They were trained on the data above.
- **Bibliography.** Books and papers consulted during the research are not included.

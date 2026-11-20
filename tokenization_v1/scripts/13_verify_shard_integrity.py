"""CLI de integridad de shards (correccion 5). Dos modos:

  --mode compute  -> calcula checksums SHA-256 de los shards YA EXISTENTES
                      y escribe outputs_v2/<test_name>/integrity_manifest.json
                      (nunca toca manifest.json ni los shards).
  --mode verify   -> relee los shards y el integrity_manifest.json existente,
                      compara, imprime PASS/FAIL.

No regenera ningun shard en ningun modo."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils_v2 import atomic_write_json  # noqa: E402
from shard_integrity import build_integrity_manifest, load_integrity_manifest, verify_shard_integrity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    ap.add_argument("--mode", choices=["compute", "verify"], required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    out_dir = os.path.abspath(os.path.join(base, config["output_dir"], args.test_name))
    integrity_path = os.path.join(out_dir, "integrity_manifest.json")

    if args.mode == "compute":
        manifest = build_integrity_manifest(shard_dir, args.test_name)
        atomic_write_json(integrity_path, manifest)
        print(json.dumps({
            "status": "computed", "test_name": args.test_name, "n_shards": manifest["n_shards"],
            "algorithm": manifest["algorithm"], "output_path": integrity_path,
        }, indent=2, ensure_ascii=False))
    else:
        if not os.path.isfile(integrity_path):
            print(json.dumps({"status": "FAIL", "reason": f"no existe {integrity_path}, correr --mode compute primero"}))
            sys.exit(1)
        manifest = load_integrity_manifest(integrity_path)
        result = verify_shard_integrity(shard_dir, manifest)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["status"] != "PASS":
            sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Strip optimizer/training state from an official AlignScore checkpoint."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "models" / "AlignScore-base" / "AlignScore-base.ckpt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "AlignScore-base" / "alignscore-base-nli.safetensors",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"checkpoint not found: {args.input}")

    import torch
    from safetensors.torch import save_file

    payload = torch.load(
        args.input,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    source_state = payload.get("state_dict", {})
    state = {
        key: value.contiguous()
        for key, value in source_state.items()
        if (
            key.startswith("base_model.")
            or key.startswith("tri_layer.")
        )
        and key != "base_model.embeddings.position_ids"
    }
    if not state or not any(key.startswith("tri_layer.") for key in state):
        raise RuntimeError("checkpoint did not contain the expected AlignScore NLI weights")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(args.input)
    save_file(
        state,
        str(args.output),
        metadata={
            "architecture": "roberta-base + AlignScore tri_layer",
            "label_order": "entailment,neutral,contradiction",
            "source_sha256": source_hash,
            "source_repository": "yzha/AlignScore",
        },
    )
    print(f"output={args.output}")
    print(f"weights={len(state)}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={file_sha256(args.output)}")
    print(f"source_sha256={source_hash}")
    return 0


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())

"""Write one MMLU parquet split as the ``rows.json`` list the prepare stage reads.

Each row keeps ``question``, ``choices``, ``answer``, and ``subject`` and gets
the id ``<split>/<row index>``. pyarrow is not a project dependency:

    uv run --with pyarrow python scripts/export_mmlu.py \
        /path/to/cais/mmlu/snapshot/all/auxiliary_train-00000-of-00001.parquet \
        /research/projects/ecoai/LLMKnowledgeRecovery/datasets/mmlu_auxiliary_train/rows.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="auxiliary_train", help="prefix for row ids")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet, columns=["question", "choices", "answer", "subject"])
    rows = [
        {"id": f"{args.split}/{index}", **row}
        for index, row in enumerate(table.to_pylist())
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

import os
os.environ["NEMO_CHECKPOINT_ASYNC"] = "False"

import nemo.collections.llm as nl
from pathlib import Path

nl.import_ckpt(
    model=nl.Qwen3Model(nl.Qwen3Config1P7B()),
    source="hf://Qwen/Qwen3-1.7B-Base",
    output_path=Path("/workspace/nemo_ckpts/qwen3-1.7b-base"),
    overwrite=True,
)
print("DONE")

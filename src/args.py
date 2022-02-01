from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgramArguments:
    run_base_path: Optional[str] = field(
        default='./runs',
        metadata={'help': 'Base path where to save runs.'}
    )

    run_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Name to identify the run and logging directory.'}
    )

    dataset_path_or_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Local path to the dataset or its name on Huggingface datasets hub.'}
    )

    lang: Optional[str] = field(
        default='javascript',
        metadata={'help': 'Programming language used in the experiments.'}
    )

    pretrained_model_path_or_name: Optional[str] = field(
        default='huggingface/CodeBERTa-small-v1',
        metadata={'help': 'Path to the pretrained language model or its name on Huggingface hub.'}
    )

    tokenizer_path_or_name: Optional[str] = field(
        default='huggingface/CodeBERTa-small-v1',
        metadata={'help': 'Path to the tokenizer or its name on the Huggingface hub.'}
    )

    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'Seed for experiments replication.'}
    )

    download_csn: bool = field(default=False, metadata={'help': 'Download CodeSearchNet dataset.'})


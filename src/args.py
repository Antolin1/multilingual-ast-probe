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

    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Local path to the dataset or its name on Huggingface datasets hub.'}
    )

    pretrained_model_name_or_path: Optional[str] = field(
        default='huggingface/CodeBERTa-small-v1',
        metadata={'help': 'Path to the pretrained language model or its name on Huggingface hub.'}
    )

    lang: Optional[str] = field(
        default='python',
        metadata={'help': 'Programming language used in the experiments.'}
    )

    lr: float = field(
        default=1e-3,
        metadata={'help': 'The initial learning rate for AdamW.'}
    )

    epochs: Optional[int] = field(
        default=20,
        metadata={'help': 'Number of training epochs.'}
    )

    patience: Optional[int] = field(
        default=5,
        metadata={'help': 'Patience for early stopping.'}
    )

    layer: Optional[int] = field(
        default=3,
        metadata={'help': 'Layer used to get the embeddings.'}
    )

    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'Seed for experiments replication.'}
    )

    download_csn: bool = field(default=False, metadata={'help': 'Download CodeSearchNet dataset.'})

    do_train: bool = field(default=False, metadata={'help': 'Run probe training.'})
    do_test: bool = field(default=False, metadata={'help': 'Run probe training.'})


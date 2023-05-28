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

    pretrained_model_name_or_path: Optional[str] = field(
        default='microsoft/codebert-base',
        metadata={'help': 'Path to the pretrained language model or its name on Huggingface hub.'}
    )

    dispatch_model_weights: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to dispatch the model weights on multiple GPUs.'}
    )

    model_weights_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the model weights if `dispatch_model_weights` set to True.'}
    )

    model_type: Optional[str] = field(
        default='roberta',
        metadata={'help': 'Architecture of the transformer model. Currently just supported t5 and roberta.'}
    )

    model_checkpoint: Optional[str] = field(
        default=None,
        metadata={'help': 'Model checkpoint directory.'}
    )

    dataset_name_or_path: Optional[str] = field(
        default='./dataset',
        metadata={'help': 'Path to the folder that contains the dataset.'}
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

    batch_size: Optional[int] = field(
        default=32,
        metadata={'help': 'Train and validation batch size.'}
    )

    patience: Optional[int] = field(
        default=5,
        metadata={'help': 'Patience for early stopping.'}
    )

    layer: Optional[int] = field(
        default=5,
        metadata={'help': 'Layer used to get the embeddings.'}
    )

    rank: Optional[int] = field(
        default=128,
        metadata={'help': 'Maximum rank of the probe.'}
    )

    orthogonal_reg: float = field(
        default=5,
        metadata={'help': 'Orthogonal regularized term.'}
    )

    hidden: Optional[int] = field(
        default=768,
        metadata={'help': 'Dimension of the feature word vectors.'}
    )

    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'Seed for experiments replication.'}
    )

    do_train: bool = field(default=False, metadata={'help': 'Run probe training.'})
    do_test: bool = field(default=False, metadata={'help': 'Run probe training.'})
    do_train_all_languages: bool = field(default=False, metadata={'help': 'Run multingual probe training with all langs.'})
    do_test_all_languages: bool = field(default=False, metadata={'help': 'Test multilingual probe with all langs.'})
    do_holdout_training: bool = field(default=False, metadata={'help': 'Run holdout training.'})

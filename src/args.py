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

    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'Seed for experiments replication.'}
    )

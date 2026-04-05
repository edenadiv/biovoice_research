"""Public workflow entry points for scripts and the CLI.

The package keeps the external surface stable while distributing the
implementation across focused modules:

- `data_prep.py` for staging and inspection
- `training.py` for branch and joint experiment execution
- `inference.py` for enrollment-conditioned scoring helpers
- `evaluation.py` for metric plots and mode comparisons
- `reporting.py` for supervisor-facing outputs
"""

from .common import ALPHA_LIMITATIONS, METRIC_INTERPRETATION_NOTES
from .data_prep import inspect_data_workflow, prepare_data_workflow
from .training import run_ablation_workflow, run_joint_workflow, run_spoof_workflow, run_sv_workflow
from .reporting import export_tables_workflow, generate_supervisor_artifacts

__all__ = [
    "ALPHA_LIMITATIONS",
    "METRIC_INTERPRETATION_NOTES",
    "export_tables_workflow",
    "generate_supervisor_artifacts",
    "inspect_data_workflow",
    "prepare_data_workflow",
    "run_ablation_workflow",
    "run_joint_workflow",
    "run_spoof_workflow",
    "run_sv_workflow",
]

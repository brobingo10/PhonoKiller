# PhonoKiller

PhonoKiller is a resumable ASE-to-Phonopy workflow that searches for a
dynamically stable periodic structure. It relaxes the input, calculates
finite-displacement force constants, follows unstable phonon modes through
q-commensurate distortions, relaxes and deduplicates the candidates, and
repeats the phonon calculation on the best structure.

## Run the workflow

Run PhonoKiller without arguments in an interactive terminal to open Mori's
guided CLI. The guide explains and validates every `run` argument, asks for the
calculator factory and optional JSON overrides for each workflow-settings
section, and builds the YAML configuration for you. No prepared YAML file is
required. Mori does not write that generated configuration, create workflow
files, or load the calculator until the run is confirmed.

```console
phonokiller
```

An incomplete `phonokiller run` command opens the same guide and presents any
supplied values as editable defaults. If `--config` already identifies a valid
YAML file, the guide can reuse it; otherwise Mori asks where to create a new
one. Complete commands remain non-interactive, which keeps scripts and batch
jobs deterministic and therefore still requires an existing YAML file:

```console
phonokiller run POSCAR --config phonokiller.yaml --output search-run
```

The optional CLI values are `--format` for an explicit ASE input format,
`--index` for an integer frame selection (default `-1`, the last frame), and
`--no-resume` to prohibit reuse of matching checkpoints. Incomplete commands
received through non-interactive input fail with exit code `2` instead of
waiting for answers.

The calculator is supplied as an importable factory. The same provider is used
for the initial relaxation, displaced-supercell forces, and candidate
relaxations. In guided mode, enter this value when Mori asks for the calculator
factory. Each following settings prompt accepts a JSON object such as
`{"max_steps": 800}` or `{}` to retain the stated PhonoKiller defaults. The
equivalent hand-written configuration for a non-interactive run is:

```yaml
calculator:
  factory: my_calculators:make_calculator
  kwargs: {}

relaxation:
  mode: full_cell
  optimizer: BFGS
  force_tolerance: 0.005
  max_steps: 500

# Unspecified values inherit from relaxation.
candidate_relaxation:
  max_steps: 800

phonopy:
  minimum_supercell_span_angstrom: 10.0
  mesh_length: 100.0

soft_modes:
  frequency_threshold_thz: -0.05
  degeneracy_tolerance_thz: 0.001
  max_mode_groups: 5
  mean_displacement_angstrom: 0.1

search:
  max_evaluations: 10
  max_candidates_per_iteration: 256
```

The finite-displacement supercell is sized automatically so all three
face-to-face spans reach the configured target. PhonoKiller passes no
workflow-defined displacement distance, primitive matrix, symmetry tolerance,
backend, or force-constant symmetrization options to Phonopy. The scalar mesh
length is passed directly to Phonopy with eigenvectors enabled.

## Outputs and resume

Each evaluation has its own directory under `iterations/`, containing its
structure, Phonopy files, soft-mode report, candidate relaxations, reduction,
and selection record. `history.json` records the complete search path. A
self-contained `final/` directory is created only after no mode lies below the
stability threshold.

Rerunning the same command resumes matching checkpoints. A changed structure,
configuration, calculator identity, or dependency version is rejected instead
of being mixed with existing results.

The Python entry point is:

```python
from phonokiller import RunConfig, run_workflow

result = run_workflow(atoms, calculator_factory, RunConfig(), "search-run")
print(result.status, result.artifacts.history)
```

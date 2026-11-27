# Research paper

**Robogogogo: An Early Robot Simulation Environment for Demonstration Data Collection**

Tian Shi, Exciton Research LLC; Ping Wang, Department of Computer Science,
Stevens Institute of Technology. Working paper dated November 27, 2026.

**[Read the paper (PDF)](robogogogo.pdf)**

The paper presents the simulation architecture, multimodal demonstration
representation, preliminary geometric characterization, and research agenda for
vision-language-action policies and learned world models. Related work includes
Google DeepMind's RT-2, Gemini Robotics, and Genie; NVIDIA's Isaac Lab, GR00T, and
Cosmos; and Physical Intelligence's pi0.5.

The manuscript reports an early environment and sampled model geometry. Learned
policy performance, world-model integration, physical fidelity, and physical robot
experiments remain future work. Engineering checks in the accompanying evidence
are supporting material, not measurements of learning utility. Local filesystem
paths in the published regression log have been replaced with `<workspace>`; test
results and timings are unchanged.

## Contents

- `robogogogo.pdf`: compiled reading copy.
- `main.tex` and `references.bib`: manuscript and bibliography.
- `figures/`: Studio screenshots with the original license and attribution.
- `evidence/`: saved geometric analyses, engineering-check records, environment
  metadata, and source provenance.
- `scripts/collect_evidence.py`: bounded evidence collection against a clean
  simulator checkout.
- `scripts/generate_results.py`: numerical LaTeX macros from the saved report.

## Build the paper

With a standard TeX installation providing `latexmk`, BibTeX, and `plainnat`, run
from this directory:

```sh
make pdf
```

This builds from the saved evidence without starting the simulator or rerunning
experiments. Temporary TeX files are ignored under `build/`; the reading copy is
updated at `robogogogo.pdf`.

## Reproduce the supporting evidence

The original evidence records software version 0.2.0, robot revision
`engineering-11`, and simulator source revision
`a886d2a9d3a27c87a9e90a21ecfbc3259657b943`. Detailed provenance and dependency
versions are in `evidence/report.json` and `evidence/sources.json`.

Prepare a Python environment containing the clean simulator checkout and its
development and Studio extras. For a checkout at the parent directory:

```sh
python3 -m venv /tmp/robogogogo-paper-env
/tmp/robogogogo-paper-env/bin/python -m pip install -e '..[dev,studio]'
/tmp/robogogogo-paper-env/bin/python scripts/collect_evidence.py --repo ..
python3 scripts/generate_results.py
make pdf
```

To reproduce the original source revision, use a separate clean checkout at the
revision above, install its extras, and pass that checkout's path to `--repo`.
Use the recorded dependency versions when reproducing the original environment;
the project's install extras specify version ranges rather than a lockfile.

The collector rejects a dirty simulator checkout, runs three selected regression
files, samples geometry, and exercises capture/save/read using synthetic images.
It does not exercise the browser renderer or learned control. The broader
seed-43 geometry analysis is historical evidence and is not rerun by this
collector. Numerical execution and JAX compilation can take several minutes.

Reproduction overwrites evidence in this directory. Review the resulting changes
before updating the manuscript. Temporary episode archives are removed after
inspection; their identifiers, timestamps, and digests vary between runs.

## License and attribution

This paper and its accompanying original materials are distributed under the
repository's [Apache-2.0 license](../LICENSE). Screenshots retain the license and
notice in `figures/`. Cited external papers are references and are not included
in this distribution.

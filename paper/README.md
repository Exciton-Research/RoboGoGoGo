# Research paper

**Robogogogo: An Early Robot Simulation Environment for Demonstration Data Collection**

Tian Shi, Exciton Research LLC; Ping Wang, Department of Computer Science,
Stevens Institute of Technology.

**[Read the paper (PDF)](robogogogo.pdf)**

The paper presents the simulation architecture, demonstration representation,
preliminary geometric characterization, and research directions for VLA policies
and learned world models.

## Build

From this directory, with a TeX installation providing `latexmk`, BibTeX, and the
packages used by `main.tex`:

```sh
make pdf
```

The build uses `main.tex`, `references.bib`, and `figures/studio.png`. It updates
`robogogogo.pdf`; temporary files remain in the ignored `build/` directory.
Simulator dependencies and experiment runs are not required.

## License

The paper and its original materials use the repository's
[Apache-2.0 license](../LICENSE). The screenshot's license and attribution are
retained in `figures/LICENSE.txt` and `figures/NOTICE.txt`.

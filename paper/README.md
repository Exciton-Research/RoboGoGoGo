# Paper

**Robogogogo: A Tabletop Simulation Workbench for Object Collection and Policy Integration**

By the Robo Team.

**[Read the paper (PDF)](robogogogo.pdf)**

The paper describes the current MuJoCo Panda environment: eight textured YCB
objects, named-target collection in a persistent tabletop scene, aligned
recording/replay, and the observation/action boundary for future VLA integration.
The scripted controller uses known scene identities and poses. The described
sequence is an illustration in a curated arrangement, not a learned-policy
benchmark or a real-robot result. RG Mobile Duo is identified as a separate,
earlier workbench.

## Build

With a TeX installation providing `latexmk`, BibTeX, and the packages in `main.tex`:

```sh
make pdf
```

The source uses `main.tex`, `references.bib`, and `figures/tabletop.jpg`. The build
updates `robogogogo.pdf`; intermediate files remain in the ignored `build/`
directory. Simulator dependencies, model downloads, and experiment runs are not
needed to build the paper.

## Assets and attribution

The figure reproduces the project's actual eight-object environment and native
RGB replay. Original project material is Apache-2.0. The pictured YCB scans remain
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); their source is the
[YCB Object and Model Set](https://ycb-benchmarks.s3.amazonaws.com/index.html),
Calli et al., Google 16k variants. These assets are distinct from the Google
Scanned Objects dataset. The Panda model comes from Google DeepMind's MuJoCo
Menagerie, under Apache-2.0. See the figure caption, references,
`figures/LICENSE.txt`, and `figures/NOTICE.txt`.

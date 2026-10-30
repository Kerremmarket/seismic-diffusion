# Conditional diffusion for seismic waveforms

**Model design and evaluation at ARMT — working paper, October 2026**

[Read the PDF](../../output/pdf/working_paper.pdf) · [Edit the LaTeX source](working_paper.tex)

The paper compares conditional and global reference diffusion with empirical nearest-neighbor generation. It reports three repetitions on 169 validation recordings, complementary waveform scores, and the effects of filtering generated traces. Validation was also used for model selection; the results are exploratory and do not establish independent-test performance.

## Included files

- `working_paper.tex`: standalone source with all four figures embedded as vector paths or plotting commands.
- `../../output/pdf/working_paper.pdf`: the compiled 15-page paper.
- `figures/waveform_draws.{png,pdf}`: observed recordings and the first three conditional draws for three events selected by magnitude rank.
- `figures/waveform_methods.{png,pdf}`: the first draw from each generation method for the same events.
- `build_exhibits.py`: regenerates four tables and two aggregate-score figures from the included score extract.
- `../analysis/paper_seeds50_20261030/CPU_RESULTS.json`: aggregate scores and uncertainty intervals, without private event-level arrays or local run paths.
- `paper_manifest.json`: SHA-256 hashes for the published files.

## Build the paper

With Tectonic available, run from the repository root:

```sh
tectonic --outdir output/pdf research/manuscript/working_paper.tex
```

The source is self-contained; compiling it requires no waveform files, model weights, external graphics or bibliography file.

To regenerate the aggregate-score exhibits using Python's standard library:

```sh
python research/manuscript/build_exhibits.py
```

This command updates only the marked numerical table and score-figure blocks. It does not regenerate the waveform samples or rerun the experiments.

## Data and experimental scope

The full waveform archive, trained models and generated ensembles are not included. The waveform examples are already embedded in the paper and available as separate figures. Their amplitudes use archive units, with a shared scale within each event column and no per-trace normalization.

Reproducing training or generating additional waveforms requires the private archive and the research implementation. The repository's root-level Stage 1 scripts describe an earlier model and do not reproduce these experiments. The included aggregate scores support rebuilding the reported exhibits, not independently recomputing the evaluation.

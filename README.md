# CPU Multimodal Recommendation Lab

### A CPU-scale FREEDOM reproduction with a missing-image study

**Train a CPU adaptation of FREEDOM on pre-extracted Amazon Baby image–text features, inspect validation rankings and five recorded development runs.** Six missing-image treatments are implemented, including the support-dependent shrinkage candidate `support_mix`.

English | [中文](README_ZH.md)

[Quick start](#quick-start) · [Current evidence](#current-evidence) · [Methods](#methods) · [Documentation](#documentation)

![Validation curves from five completed development runs; these are not test results](results/interim/validation_curves.png)

*Validation Recall@20 across five completed development runs on one fixed Baby subset. Curves from different conditions are shown separately; this figure is not a paired method comparison.*

![cpu-multimodal-rec-lab](results/interim/cartoon-infographic.png)

## Current evidence

The **implementation and interim report** include five completed development configurations out of 16: four `clean` backbone candidates and one `uniform30` fixed-mix candidate. A sixth run stopped after a saved epoch-35 checkpoint. The 45-run main study, independent test evaluation and comparative evaluation of `support_mix` remain pending.

On the prepared subset (1,993 items, 8,967 users), the selected `clean` backbone reached **13.3173% validation Recall@20** and **5.9899% validation NDCG@20** at epoch 85. This development result was selected on validation with seed 101 on the prepared Baby subset. The five complete runs took 7.73–18.53 minutes each and observed 660.11–750.90 MiB peak process RSS on an i7-12700H with two CPU training threads. See the [interim findings](docs/findings.md) and [machine-readable summary](results/interim/development_summary.csv).

The five saved best checkpoints were independently loaded and their validation Top-20 rankings recomputed for 2,803 users; the comparison matched the original records. [Verification metadata](results/interim/verification.json) documents this historical check. **Raw data, prepared datasets, checkpoints (`best.pt` / `resume.pt`), per-user results, and local environments are not distributed in this repository.** Therefore the published summary is inspectable, but rerunning checkpoint verification requires regenerating local data and training outputs.

## Quick start

Requires Python 3.12 and Git. Commands below use PowerShell; model training and evaluation run on CPU. The published dependency lock records the tested versions, including PyTorch `2.14.0+cpu`. The pip installation route below has not been separately tested on another machine; the recorded local environment was installed with `uv`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
git clone https://github.com/enoche/MMRec.git external/MMRec
git -C external/MMRec checkout c68ee7c9b0de1ca7c5150258322edb4994be29d9
.\.venv\Scripts\python.exe -m mmrec_lab download
.\.venv\Scripts\python.exe -m mmrec_lab prepare --items 2000 --output data/processed/baby2000
.\.venv\Scripts\python.exe -m mmrec_lab environment
```

The downloader obtains the five author-hosted Baby interaction, feature, and mapping files; it verifies size, format, and local SHA256. The target of 2,000 items becomes 1,993 after training-only pruning in the recorded preparation. Source links, IDs, hashes, and data-license limits are in [data provenance](docs/data-provenance.md). Availability of the external Google Drive files may change.

To run correctness checks against the pinned upstream implementation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The historical pre-main integration run reported 76 passed tests plus 30 passing subtests (106 JUnit cases). This is historical evidence, not a claim that a fresh installation was tested here. Training, development search, main-study, reporting, and benchmark commands are exposed by `python -m mmrec_lab --help`; they create local outputs and are not required to read this repository's interim report. `scripts/export_interim.py` requires the **five local run directories and checkpoints**, which are not published; it cannot reconstruct them from the summary CSV.

## Methods

FREEDOM uses fixed image–text item graphs and a denoised user–item interaction graph. The CPU adaptation follows the pinned [MMRec FREEDOM implementation](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/src/models/freedom.py), with documented differences in the [reproduction map](docs/reproduction-map.md).

For artificially hidden image vectors, the candidate uses up to 20 training co-interaction neighbors. With `n` visible neighbors, visible global image mean `μ`, and validation-selected `τ > 0`:

```text
support_mix = (sum(visible neighbor vectors) + τ × μ) / (n + τ)
```

It falls back to `μ` when `n = 0`; hidden values and imputed vectors do not provide evidence to other targets. The implementation also includes `zero_fill`, `global_mean`, `paper_neighmean`, `observed_mean`, and `fixed_mix`. The planned `uniform30` and `tail30` scenarios hide the same number of item image vectors while retaining text and interactions. Prior graph imputation and shrinkage work limits any novelty claim; see [related work](docs/related-work.md). The author's preprocessed features may already include mean-filled native missing images, so `clean` means only *no additional artificial mask*.

## Documentation

- [Interim findings](docs/findings.md): five validation results, resource observations, verification scope, and unresolved claims.
- [Data provenance](docs/data-provenance.md): author-hosted files, local hashes, subset identity, and redistribution boundary.
- [Reproduction map](docs/reproduction-map.md), [paper notes](docs/paper-reading.md), and [related work](docs/related-work.md).
- [Support-count diagnostics](docs/support-diagnostics.md): training-graph statistics, not recommendation effectiveness.
- [Implementation notes](docs/implementation.md): evaluation, checkpoint identity, and leakage controls.

## Attribution and license

The adapted model is based on Xin Zhou's MMRec implementation and retains its GPL-3.0 attribution; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The separate FREEDOM repository has a different license statement. The code license does not grant redistribution rights to author-preprocessed Amazon data. This repository does not include those data or the original images/text.

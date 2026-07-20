# Post-Stimulus Choice Decoding Is Stronger From Regional Firing Rates Than From Inter-Area Correlations

Reanalysis of the Steinmetz et al. (2019) Neuropixels dataset comparing two **matched-dimensional feature families** — regional mean firing rates vs. trial-wise inter-area correlations — for decoding perceptual choice.

**Preprint (Zenodo):** https://doi.org/10.5281/zenodo.21459798

## Key Finding

Using balanced accuracy, leakage-resistant preprocessing (all fitting done inside training folds), dimension-matched feature families, and mouse-level inference (n = 10 mice):

- **Before stimulus onset:** neither regional firing rates (49.5%) nor inter-area correlations (50.1%) decode choice above chance.
- **After stimulus onset:** regional firing-rate features are substantially more predictive than matched-dimensional correlation features (0–250 ms: 64.2% vs 53.4%; 250–500 ms: 67.3% vs 50.1%; Holm-adjusted paired *p* = 0.0078).

The supported result is a **time-dependent difference in decodability between two feature representations** — *not* a causal switch from distributed to local computation. Post-stimulus regional-rate decoding scales with stimulus discriminability and is much stronger on correct than error trials (a substantial stimulus-linked component), yet above-chance decoding on equal-contrast trials shows that stimulus side alone cannot explain it.

## Results Summary (mouse-level balanced accuracy)

| Time Window | Regional rates | Inter-area corr. | Difference | Holm *p* |
|---|---|---|---|---|
| Pre-early (−500 to −250 ms) | 49.2% | 50.3% | −1.1% | 0.609 |
| Pre-late (−250 to 0 ms) | 49.5% | 50.1% | −0.6% | 0.609 |
| Post-early (0 to +250 ms) | **64.2%** | 53.4% | +10.9% | 0.0078 |
| Post-late (+250 to +500 ms) | **67.3%** | 50.1% | +17.2% | 0.0078 |

## Repository Structure

```
.
├── index.html                        # Revised manuscript (preprint)
├── reanalysis_results_summary.json   # Summary of the reanalysis outputs
├── requirements.txt                  # Package dependencies
└── PY/
    ├── analysis_additional.py        # Revised analysis script (fixed seed)
    ├── cross_area_communication.py
    ├── cross_area_robust_analysis.py
    ├── pre_vs_post_analysis.py
    └── ...                           # Earlier analysis scripts
```

## Data

Publicly available Steinmetz et al. (2019) dataset:
- **Source:** https://figshare.com/articles/dataset/steinmetz/9598406
- **Reference:** Steinmetz NA, Zatka-Haas P, Carandini M, Harris KD (2019). Distributed coding of choice, action and engagement across the mouse brain. *Nature* 576:266–273.

Download the data and place the `.npy` files in a local `DATA/` folder (not included here).

## Requirements

```bash
pip install -r requirements.txt
```
(numpy, scipy, scikit-learn, matplotlib, pandas)

## Citation

- **Preprint:** Sivroni S (2026). *Post-Stimulus Choice Decoding Is Stronger From Regional Firing Rates Than From Inter-Area Correlations.* Zenodo. https://doi.org/10.5281/zenodo.21459798
- **Dataset:** Steinmetz NA, Zatka-Haas P, Carandini M, Harris KD (2019). *Nature* 576:266–273.

## License

Code: MIT License. Manuscript: CC BY 4.0.

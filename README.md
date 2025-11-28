# Temporal Dissociation in Neural Decision Signals

Analysis of inter-area correlations vs. local activity in perceptual decision-making using the Steinmetz et al. (2019) Neuropixels dataset.

## Key Finding

We found a **temporal dissociation** in choice-predictive neural signals:
- **Pre-stimulus**: Both inter-area correlations and local activity predict choice equally (~54%)
- **Post-stimulus**: Local activity dominates (64-69%), while inter-area correlations remain modest (55-57%)

This demonstrates a transition from distributed network states to localized evidence coding as sensory information becomes available.

## Project Structure

```
.
├── PY/                          # Python analysis scripts
│   ├── cross_area_communication.py    # Original cross-area correlation analysis
│   ├── cross_area_robust_analysis.py  # Robust analysis with all controls
│   └── pre_vs_post_analysis.py        # Pre vs post-stimulus comparison
├── DATA/                        # Steinmetz dataset (not included, see below)
├── FIGURES/                     # Generated figures
│   ├── cross_area_results.png
│   ├── robust_analysis_results.png
│   └── pre_vs_post_analysis.png
├── MANUSCRIPTS/                 # Paper drafts
│   ├── manuscript_draft.md           # Original draft
│   └── manuscript_revised.md         # Revised with new findings
└── README.md
```

## Data

The analysis uses the publicly available Steinmetz et al. (2019) dataset:
- **Source**: https://figshare.com/articles/dataset/steinmetz/9598406
- **Reference**: Steinmetz NA, Zatka-Haas P, Carandini M, Harris KD (2019). Distributed coding of choice, action and engagement across the mouse brain. Nature 576:266-273.

Download the data and place the `.npy` files in the `DATA/` folder.

## Requirements

```
numpy
scipy
scikit-learn
matplotlib
pandas
xgboost (optional)
```

Install with:
```bash
pip install numpy scipy scikit-learn matplotlib pandas xgboost
```

## Usage

### 1. Original Cross-Area Analysis
```bash
cd PY
python cross_area_communication.py
```

### 2. Robust Analysis with Controls
Includes: confound regression, cross-mouse validation, partial correlations, multiple classifiers
```bash
python cross_area_robust_analysis.py
```

### 3. Pre vs Post-Stimulus Comparison
Main analysis comparing inter-area and local signals across time
```bash
python pre_vs_post_analysis.py
```

## Results Summary

| Time Window | Inter-Area | Local | p-value |
|-------------|------------|-------|---------|
| Pre (-250 to 0ms) | 53.5% | 54.7% | 0.80 (n.s.) |
| Post (0 to +250ms) | 57.2% | **64.5%** | <0.0001 |
| Post (+250 to +500ms) | 54.9% | **69.0%** | <0.0001 |

## Citation

If you use this code, please cite:
- Steinmetz NA, Zatka-Haas P, Carandini M, Harris KD (2019). Distributed coding of choice, action and engagement across the mouse brain. Nature 576:266-273.

## License

MIT License

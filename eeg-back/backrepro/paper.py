"""Numbers reported in Shafi et al. 2026 (Sci Data, doi:10.1038/s41597-026-07607-3)."""

# Fig. 5: Fp1-A1, Welch, n_fft 2048, 1-40 Hz. Peak in Hz and mean power relative to nw.
FIG5 = {"nw": (10.16, 1.00), "lw": (1.37, 0.42), "mw": (2.54, 0.75), "hw": (2.34, 1.49)}

# Fig. 6: clean vs noisy, n_fft 4096. Peaks in Hz and noisy/clean mean-power ratio.
FIG6 = {"clean_peak": 2.15, "noisy_peak": 1.37, "ratio": 4.01}

# Fig. 8 colorbar limits, each shared by the workload and no-workload maps of a row and band.
FIG8_COLORBAR = {("clean", "Theta"): (1.0, 14.3), ("clean", "Alpha"): (0.8, 11.6),
                 ("noisy", "Theta"): (0.4, 5.8), ("noisy", "Alpha"): (1.0, 15.5)}

# Repeated-measures ANOVA on frontal theta, 30 clean subjects, lw / mw / hw.
ANOVA = {"F": 4.21, "df": (2, 58), "p": 0.02,
         "posthoc": {("lw", "mw"): 0.12, ("lw", "hw"): 0.02, ("mw", "hw"): 0.19}}

# Table 3: accuracy, precision, recall, F1 in %.
TABLE3 = {
    ("binary", "clean", "MLP"): (83.61, 83.73, 83.60, 83.59),
    ("binary", "clean", "XGBoost"): (87.48, 87.65, 87.48, 87.47),
    ("binary", "noisy", "MLP"): (71.33, 71.35, 71.33, 71.32),
    ("binary", "noisy", "XGBoost"): (78.09, 78.12, 78.09, 78.09),
    ("ternary", "clean", "MLP"): (79.83, 80.17, 79.83, 79.70),
    ("ternary", "clean", "XGBoost"): (82.80, 82.98, 82.80, 82.66),
    ("ternary", "noisy", "MLP"): (64.01, 64.13, 64.01, 64.03),
    ("ternary", "noisy", "XGBoost"): (69.18, 69.39, 69.17, 69.21),
}

"""Numbers reported in the paper, for the paper-vs-ours comparison."""

# Tables 1 and 2: classifier -> (AC, SE, SP)
OVERALL = {
    "mat": {"Tree": (94.6, 95, 94), "Discriminant": (90.5, 89, 92),
            "Naive Bayes": (91.8, 92, 91), "KNN": (96.2, 97, 96),
            "SVM": (95.3, 94, 96), "EL": (97.4, 97, 98)},
    "stew": {"Tree": (95.2, 97, 94), "Discriminant": (91.8, 93, 91),
             "Naive Bayes": (94.4, 95, 94), "KNN": (95.9, 96, 97),
             "SVM": (94.7, 97, 92), "EL": (96.1, 96, 97)},
}

# Tables 3 and 4: lead -> (AC, SE, SP)
LEADWISE = {
    "mat": {"Fp1": (91.1, 91, 91), "Fp2": (93.9, 94, 94), "Fz": (90.7, 90, 91),
            "F3": (94.5, 93, 96), "F4": (92.9, 92, 94), "F7": (89.4, 88, 91),
            "F8": (90.0, 88, 92), "T3": (90.6, 91, 90), "T4": (80.1, 79, 81),
            "T5": (86.8, 85, 89), "T6": (88.1, 99, 77), "P3": (88.2, 87, 89),
            "P4": (89.1, 89, 90), "Pz": (89.0, 89, 89), "O1": (90.5, 93, 88),
            "O2": (64.5, 84, 45), "C3": (89.8, 92, 88), "C4": (87.3, 91, 83),
            "Cz": (88.9, 88, 90)},
    "stew": {"AF3": (90.4, 91, 89), "F7": (92.9, 92, 94), "F3": (94.0, 93, 95),
             "FC5": (90.2, 90, 90), "T7": (90.7, 90, 92), "P7": (88.4, 89, 88),
             "O1": (90.5, 90, 91), "O2": (90.5, 93, 88), "P8": (87.2, 99, 75),
             "T8": (88.2, 86, 90), "FC6": (89.3, 90, 89), "F4": (93.4, 94, 93),
             "F8": (89.5, 89, 91), "AF4": (90.7, 90, 92)},
}

# Fig. 9, read off the plot: accuracy against decomposition level (at 100 BAO
# iterations) and against BAO iterations (at L = 5), MAT.
TUNING = {
    "overall": {"level": {3: 94.5, 5: 97.4, 7: 96.1, 9: 95.2},
                "bao_iterations": {25: 93.6, 50: 95.2, 75: 96.2, 100: 97.4}},
    "leadwise": {"level": {3: 92.2, 5: 94.5, 7: 93.0, 9: 92.7},
                 "bao_iterations": {25: 91.2, 50: 92.2, 75: 93.2, 100: 94.5}},
}

# Fig. 3, read off the plot: BAO fitness at the first and last iteration.
FITNESS = {"mat": (0.0555, 0.037), "stew": (0.055, 0.0295)}

"""Joining the Shin 2018 and COG-BCI n-back EEG datasets into one feature table.

remote      range reads of the source zip archives
sources     dataset locations, versions, channel sets, label maps
loaders     per-dataset readers returning a common Recording
preprocess  the harmonization pipeline (filter, resample, ICA, reference, channels)
features    windowing and spectral features
"""

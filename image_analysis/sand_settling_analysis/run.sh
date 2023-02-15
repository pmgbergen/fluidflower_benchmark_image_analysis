#!/bin/bash

#The sand settling analysis needs careful preparation in order to
#reduce the effects of camera movements. Therefore, a preparation
#step is performed before performing the compaction analysis.

#Thus first run:
python 1_preparation.py

#In addition, the water zone incl the color checker shall not be used for the analysis.
#Thus, some masks have to be detected. Run:
python 2_read_labels.py

#Then, depending on what runs should be compared choose the correct
#flag '', and run:
python 3_analysis.py

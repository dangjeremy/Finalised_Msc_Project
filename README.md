# Finalised MSc project

This MSc project has several files such as data processing and modelling. The data used for the project has been excluded for data privacy.

There are several files:

Those with prefix "GCMC_" refers to current (as of 18th Sep 2019) state-of-the-art implementation of a version of Graph Convolutional Neural Networks (specifically Graph Convolutional Matrix Completion)

Files with prefix "CF_" refers to traditional collaborative filtering methods.

--------------------------
main_data_processing.ipynb
--------------------------

Contains the self created data pipeline:

1) Reading files from raw unprocessed data.
2) Performing data cleansing on text data
3) Perform summarisation using tf-idf
4) Transformation of data to obtain the correct format for a recommendation system
5) Transformation of data to obtain a graph format for SoTA methods

--------------------------

Models' used are:

1) NMF
2) Graph Convolutional Neural Networks

## Lexical Analysis

This directory contains the code and data used for the lexical analysis.
It is organized as follows:

* `analysis_main` explores broad patterns based on log-likelihood and
  neighborhood density values for individual words
* `analysis_clusters` analyzes the clustering patterns based on contextualized
  embeddings for individual words
* `analysis_ngrams` applies the log-likelihood analysis to word 5-grams

The Jupyter notebooks rely on the following data sources:
* consolidated frequency and log-likelihood information (`freq` files)
* neighborhood density information (`avg_density` file)
* word-level cluster outputs (in `cluster_out/` directory)

The frequency information is available for the naturalistic corpus (`cacl_t1`)
and the LLM-improved corpus (`sample_gpt35`).

For space reasons, we are currently unable to include the 5-grams file for the
naturalistic corpus in this repository.
For the same reason, we are only including cluster outputs for the individual
examples discussed in the paper.
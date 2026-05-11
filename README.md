## Analyzing Changes in Scientific Communication

This repository contains the code for the following paper:

> Filip Miletić \& Neele Falk. **What are LLMs doing to scientific communication? Measuring changes in writing practices and reading experience.** In *Proceedings of the 15th Conference on Language Resources and Evaluation (LREC 2026)*, Palma de Mallorca, Spain. <https://doi.org/10.63317/3ai7wig4fhd8>

Below, we provide a general overview of this repository. Each subdirectory
additionally contains a readme file with more detailed information.


### ACL Anthology Corpus
Our paper uses a corpus of papers from the ACL Anthology, which is effectively
an update of a previously released corpus. The full pipeline to update the
corpus is available in the `update_original_corpus` directory.

### Data Preparation
The code used for further data preparation is available in the `prep_data`
directory. This includes general preprocessing, computation of frequency-based
lexical information, and implementation of a topic model, word2vec models,
and clustering over contextualized word embeddings.

### Lexical Analysis
The code for the analysis of lexical patterns, together with dataframes
consolidating the word-level information produced in the previous step,
is provided in the `lexical_analysis` directory.

### Feature Analysis
The code for the analysis of linguistic features (extraction, feature selection and regression) can be found in the `feature_analysis` folder.
The results of the feature selection process and the regression analysis can be found in that folder under `results_feature_analysis` folder. 

### Data
All data used for the analysis can be found in the `data` folder. The anonymized human pairwise preference annotations can also be found in the `data` folder.
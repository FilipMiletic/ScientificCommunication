TODO:
```
[ ] Go over directories, clean them up + add READMEs  
    [ ] prep_data  
    [ x] data (contains the LLM and original dataset and the human annotation data (+ the full potato configuration which includes all instructions, consent form etc)
    [ x] feature analysis (contains also all results)
    [ ] update_original_corpus  
    [ ] top-level  
[ ] Make repository public  
```
---
This repository contains the code for the following paper:

> Filip Miletić \& Neele Falk. **What are LLMs doing to scientific communication? Measuring changes in writing practices and reading experience.** To appear in *Proceedings of the 15th Conference on Language Resources and Evaluation (LREC 2026)*, Palma de Mallorca, Spain.

### Feature Analysis
The code for the analysis of linguistic features (extraction, feature selection and regression) can be found in the `feature_analysis` folder.
The results of the feature selection process and the regression analysis can be found in that folder under `results_feature_analysis` folder. 
A detailed readme with the information about the feature selection process and the results can be found in `results_feature_analysis/readme.md`.

### Data
All data used for the analysis can be found in the `data` folder. The anonymized human pairwise preference annotations can also be found in the `data` folder.
A detailed readme with the information about the data can be found in `data/readme.md`.

# Narrative comprehension encoding models with Whisper and SBERT
This repository contains code for predicting brain activity during narrative comprehension using deep learning model embeddings (Whisper and SBERT). We compare acoustic-phonetic features from speech recognition models with semantic sentence representations to understand how different linguistic representations capture neural processes during story listening. The analysis uses encoding models with cross-narrative prediction to evaluate generalization across four spoken narratives from the [Narratives](https://snastase.github.io/datasets/ds002345) dataset ([Nastase et al., 2021](https://doi.org/10.1038/s41597-021-01033-3)), comprising $N = 47$ subjects per narrative (`black`, `forgot`, `piemanpni`, `bronx`).

#### Workflow
1. Use `clean.py` to perform confound regression on fMRI data (motion parameters, aCompCor, word onset/rate).
2. Use `convert_clean_h5_to_pkl.py` to convert cleaned voxel-wise data to parcellated format using Schaefer atlases.
3. Use `whisper_feature.py` or `whisper_feature_sa.py` to extract Whisper embeddings (acoustic-phonetic and linguistic features).
4. Use `sbert_feature.py` or `sbert_feature+1.py` to extract SBERT embeddings (cumulative sentence representations).
5. Use `whisper_average_encoding.py` to run encoding analysis with Whisper features using cross-narrative prediction.
6. Use `sbert_encoding.py` to run encoding analysis with SBERT features using cross-narrative prediction.
7. Use `encoding_combined.py` to run encoding analysis combining both Whisper and SBERT features.

#### References
- Nastase, S. A., Liu, Y.-F., Hillman, H., Zadbood, A., Hasenfratz, L., Keshavarzian, N., Chen, J., Honey, C. J., Yeshurun, Y., Regev, M., Nguyen, M., Chang, C. H. C., Baldassano, C., Lositsky, O., Simony, E., Chow, M. A., Leong, Y. C., Brooks, P. P., Micciche, E., Choe, G., Goldstein, A., Vanderwal, T., Halchenko, Y. O., Norman, K. A., & Hasson, U. (2021). The "Narratives" fMRI dataset for evaluating models of naturalistic language comprehension. *Scientific Data*, *8*, 250. https://doi.org/10.1038/s41597-021-01033-3
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. In *International Conference on Machine Learning* (pp. 28492–28518). PMLR. https://arxiv.org/abs/2212.04356
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). https://arxiv.org/abs/1908.10084
- Dupré la Tour, T., Eickenberg, M., Gallant, J. L., & Nunez-Elizalde, A. O. (2022). Feature-space selection with banded ridge regression. *NeuroImage*, *264*, 119728. https://doi.org/10.1016/j.neuroimage.2022.119728

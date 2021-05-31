# GammaCas

### Directory Structure:

1. Code/: contains GammaCasModel.py, implementation for GammaCas
2. Data-GammaCas/: contains 3 .csv, .pkl data files used by GammaCasModel.py </br>
   * NewDataTest_news.csv
   * tweetSeqTest_news_.pkl
   * tweetSeqTrain_news_.pkl

   The other four files are present here inside Data-GammaCas-part2/: https://drive.google.com/drive/folders/1syrMVFhnXxn6wzMOZ901DPy52S7xAC1m?usp=sharing </br>
   * NewDataTrain_news.csv
   * newsTrainSeq.pkl
   * newsTestSeq.pkl
   * tweetNewsVocab_word2vec.pkl
3. AblationVariants/: contains three ablation variants of GammaCas

### Baselines Used:

1. TiDeH: https://github.com/NII-Kobayashi/TiDeH
2. SEISMIC: https://cran.r-project.org/web/packages/seismic/index.html
3. DeepHawkes: https://github.com/CaoQi92/DeepHawkes
4. DeepCas: https://github.com/chengli-um/DeepCas
5. ChatterNet: https://github.com/LCS2-IIITD/ChatterNet

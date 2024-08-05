# FAKEnHATE
Fake news detection in Spanish

1. Baseline: simple RoBERTa model with the best parameters.
    - Learning rate: 2e-5
    - Batch size: 8
    - Learning rate scheduler: linear (0.7-1)
2. To mask named entities:
    - The spanish entity recognizer has identified four types of NE: 
        - person (PER): 12099
        - organization (ORG): 5482
        - places (LOC): 12438
        - miscellanea (MISC):12170
    - We have perform an ablation study, masking different combinations of NE. The best results have been produced by masking the person and organization entities. We have adjusted the training parameters with the following quantities:
        - Learning rate: 5e-6
        - Batch size: 16
        - Learning rate scheduler: linear (0.6-1)
3. Add new data to the corpus:
    - Fake or unreliable news from https://hrashkin.github.io/factcheck.html, hoax labelled samples. This dataset doesn't contain any title so, to avoid biases, we have artifically generated them by using a Flan-T5 model.
    - Reliable news from https://osf.io/snpcg/ (LOCO dataset)
    For both sources we have translated the text and headline with the English to Spanish Marian model(Helsinki-NLP/opus-mt-es-en). Then, we have progressively introduced folds of 2000 samples (50% each category) in our training dataset and analysed the results. Also, for each data source we have selected the most informative data. To get that score we have obtained the embedding of the text of each article and calculated the mean of all of them. Then we have given a probability of being selected proportional to the distance to that center. Thus we have selected the most different articles.
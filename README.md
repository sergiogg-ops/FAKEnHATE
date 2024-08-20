# FAKEnHATE
Fake news detection in Spanish

1. Baseline: simple RoBERTa model with the best parameters.
    - Learning rate: 2e-5
    - Batch size: 8
    - Learning rate scheduler: linear (0.7-1)
2. To mask named entities:
    - The original dataset had already masked numbers, phones and URLs.
    - The spanish spacy entity recognizer has identified four types of NE: 
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
    
    For both sources we have translated the text and headline with the English to Spanish Marian model(Helsinki-NLP/opus-mt-es-en). Then, we have progressively introduced folds of samples (50% each category) in our training dataset and analysed the results. Also, for each data source we have selected the most informative data with the following approaches:
    - Randomly select samples.
    - Select the most different samples from each source: using the embeddings we gave more probability of being chosen to the samples more distant from the center of that dataset.
    - Select the samples that would make our dataset more diverse: using the embeddings we gave more probability of being chosen to the samples more distant to the center of our dataset.
4. Apply backtranslation to the original corpus:
    | ID     | Pipe        |
    |--------|-------------|
    | pipe0  | es-de-zh-es |
    | pipe1  | es-zh-de-es |
    | pipe2  | es-hi-ko-es |
    | pipe3  | es-af-fa-es |
    | pipe4  | es-ja-fr-es |
    | pipe5  | es-sv-zh-es |
    | pipe6  | es-fi-el-es |
    | pipe7  | /es-ru-ar-es |
    | pipe8  | es-fr-ko-es |
    | pipe9  | es-el-af-es |
    | pipe10 | es-ru-hi-es |
    | pipe11 | es-ko-af-es |
5. Use noisy embeddings to train the model:
    - Uniform random noise with scale factor of 10 (20)
    - Gaussian random noise with scale factor of 0.05
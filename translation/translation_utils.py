from tqdm import tqdm
from transformers import TranslationPipeline, logging
from nltk.tokenize import sent_tokenize

class TranslationPipelineWithProgress:
    '''
    Creates an envelope for the TranslationPipeline that shows a progress bar.
    '''
    def __init__(self, model,tokenizer, batch_size, device, **kwargs):
        '''
        Initializes the TranslationPipelineWithProgress.

        Args:
            model: Model to use for translation.
            tokenizer: Tokenizer to use for translation.
            batch_size: Size of the batches.
            device: Device to use for translation.
        
        Returns:
            TranslationPipelineWithProgress object.
        '''
        self.translator = TranslationPipeline(model=model,tokenizer=tokenizer, batch_size=batch_size, device=device, **kwargs)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __call__(self, texts, desc=None, **kwargs):
        '''
        Translate the texts using the TranslationPipeline.

        Args:
            texts: List with the texts to translate.
            **kwargs: Additional arguments for the TranslationPipeline.
        
        Returns:
            List with the translations.
        '''
        if not desc:
            desc = f"Traduciendo {kwargs['src_lang']}-{kwargs['tgt_lang']}"
        translations = []
        logging.set_verbosity_error()
        for i in tqdm(range(0, len(texts), self.batch_size), desc=desc, unit="batch"):
            batch_texts = texts[i:i + self.batch_size]
            batch_translations = self.translator(batch_texts, **kwargs)
            translations.extend(batch_translations)
        logging.set_verbosity_warning()
        return translations

def split_lists(data, field = 'message_description'):
    '''
    Split the lists in the data into sentences and mark the first sentence of each sample.

    Args:
        data: DataFrame with the data.

    Returns:
        sents: List with the sentences.
        mask: List with the mask.
    '''
    sents, mask = [], []
    for i, sample in data.iterrows():
        curr_sents = sample[field]
        if len(curr_sents) == 0:
            sents.append('')
            mask.append(-1)
        else:
            sents += curr_sents
            mask += [True] + [False] * (len(curr_sents)-1)
    return sents, mask

def split_texts(data, field = 'message_description'):
    '''
    Split the texts in the data into sentences and mark the first sentence of each sample.

    Args:
        data: DataFrame with the data.

    Returns:
        sents: List with the sentences.
        mask: List with the mask.
    '''
    sents, mask = [], []
    for _, sample in data.iterrows():
        if not sample[field]:
            sents.append('')
            mask.append(-1)
            continue
        curr_sents = sent_tokenize(sample[field])
        if len(curr_sents) == 0:
            sents.append('')
            mask.append(-1)
        else:
            sents.extend(curr_sents)
            mask.extend([True] + [False] * (len(curr_sents)-1))
    return sents, mask

def wrapp_texts(sents, mask):
    '''
    Wrapp the sentences into the original samples.

    Args:
        sents: List with the sentences.
        mask: List with the mask.
    
    Returns:
        data: List with the samples.
    '''
    data = []
    for sent, start in zip(sents, mask):
        if start == -1:
            data.append('')
        elif start:
            data.append(sent)
        else:
            data[-1] += ' ' + sent
    return data

def wrapp_lists(sents, mask):
    '''
    Wrapp the sentences into the original samples.

    Args:
        sents: List with the sentences.
        mask: List with the mask.
    
    Returns:
        data: List with the samples.
    '''
    data = []
    for sent, start in zip(sents, mask):
        if start == -1:
            data.append([])
        elif start:
            data.append([sent])
        else:
            data[-1].append(sent)
    return data

def translate(translator, data, src_lang, tgt_lang, field='message_description'):
    '''
    Translate the captions of the samples in the data.

    Args:
        translator: TranslationPipeline to use.
        data: DataFrame with the data.

    Returns:
        translations: List with the translations.
    '''
    contains_lists = any([isinstance(sample[field], list) for _, sample in data.iterrows()])
    if contains_lists:
        texts, mask = split_lists(data, field=field)
    else:
        texts, mask = split_texts(data, field=field)
    output = translator(texts, max_length=translator.tokenizer.model_max_length, return_text=True, truncation=True, src_lang=src_lang, tgt_lang=tgt_lang)
    output = [out['translation_text'] for out in output]
    if contains_lists:
        translations = wrapp_lists(output, mask)
    else:
        translations = wrapp_texts(output, mask)
    return translations
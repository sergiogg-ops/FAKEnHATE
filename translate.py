import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, TranslationPipeline
from argparse import ArgumentParser
import translation_utils as utils
from iso639 import Lang
import numpy as np
import os

def read_parameters():
    parser = ArgumentParser(description='Translates the content of a field and stores it in another field into the output file.')
    parser.add_argument('file', help='Path to the input file.')
    parser.add_argument('-if', '--input_field', default='message_description', help='Field to translate.')
    parser.add_argument('-of', '--output_field', default='translated_message', help='Field to store the translation.')
    parser.add_argument('-out', '--output', help='Output file, by the default the input file.')
    parser.add_argument('-a','--append',action='store_true',help='Whether to append the augmented data to the output file, or to overwrite it.')
    parser.add_argument('-b','--batch_size',type=int,default=16,help='Batch size for the translation.')
    parser.add_argument('-v','--verbose',action='store_true',help='Whether to show a progress bar or not.')
    args = parser.parse_args()
    return args

def main():
    args = read_parameters()
    data = pd.read_json(args.file)

    pipeline = utils.TranslationPipelineWithProgress if args.verbose else TranslationPipeline
    num_workers = os.cpu_count() - 1
    translator = pipeline(model=MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-es"),
                                        tokenizer=MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-es"),
                                        device=0,
                                        batch_size=16,
                                        num_workers=num_workers)

    translation = utils.translate(translator, data, 'en', 'es',field=args.input_field, flag=args.output_field)
    for i in range(len(translation)):
        if isinstance(translation[i],list) and len(translation[i]) == 0:
            translation[i] = [np.nan]
    translation = np.array(translation,dtype=object)
    data[args.output_field] = translation

    mode = 'a' if args.append else 'w'
    filename = args.output if args.output else args.file
    data.to_json(filename, orient='records', mode=mode)

if __name__ == '__main__':
    main()
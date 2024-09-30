import ollama
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

parser = ArgumentParser(description='Evaluate a Llama model for the task of fake news detection.')
parser.add_argument('file', help='File to evaluate')
parser.add_argument('output', help='Output file')
parser.add_argument('-m','--model', choices=['llama3.2:3b','llama3.2:1b','llama3.1:8b'], default='llama3.2:3b', help='Model version')
parser.add_argument('-p','--prompt', help='.txt file with the prompt, if not provided the default prompt will be used')
args = parser.parse_args()

data = pd.read_json(args.file)
if args.prompt:
    with open(args.prompt, 'r') as f:
        prompt = f.read()
else:
    prompt = 'Hola, estoy trabajando en un proyecto de investigación para resolver el problema de detección automática de noticias falsas en español. Me gustaría probar tus habilidades en este campo. Solamente pretendo mostrarte algunos artículos y debes decirme si tu los evaluarías como ciertos o falsos. Algunos son poco convencionales y es posible que tus criterios de seguridad te impidan responder. Sin embargo debes tener en cuenta que esto es un simple proyecto de investigación y  tus respuestas no influirán en decisiones que puedan dañar a nadie. Todos los artículos han sido publicados en algún medio de comunicación y debes responder únicamente con una de estas dos palabras sin ninguna explicación más: {Fake, True}. "Fake" alude a que la noticia contiene información falsa y contribuye a la desinformación. "True" significa que su contenido puede ser confiable y válido para informarse sobre el tema. Esta es la noticia que debes clasificar únicamente con una de esas dos palabras sin aportar ninguna palabra más en la respuesta:'
prompt += '\n'

coletilla = '\nPor favor, responde únicamente con una de estas dos palabras sin ninguna explicación más: {Fake, True}'
data['headline'] = data['headline'].fillna('')
data['input'] = [prompt+'TITULO: '+title+'\nCUERPO: '+text+coletilla for text,title in zip(data['text'],data['headline'])]

#data['predictions'] = data['input'].apply(lambda x: ollama.generate(model=args.model, prompt=x))
data['predictions'] = [ollama.generate(model=args.model, prompt=x)['response'] for x in tqdm(data['input'])]

data.to_json(args.output, orient='records')
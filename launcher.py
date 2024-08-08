import subprocess

lang = ['hi','af','ja','sa','fi','ru','fr','el','la','ko']

for i,l in enumerate(lang):
    subprocess.run(['python','translate_plus.py','data/base/train.json','-l','es',l,'es','-if','text','-of','text','-out',f'data/backtranslation/pipe{i+2}.json','-v'])
    subprocess.run(['python','translate_plus.py',f'data/backtranslation/pipe{i+2}.json','-l','es',l,'es','-if','headline','-of','headline','-out',f'data/backtranslation/pipe{i+2}.json','-a','-v'])
import subprocess

lang = ['de','zh','hi','af','ja','sa','fi','ru','fr','el','la','ko']
ofset = 0

for i,l in enumerate(lang):
    subprocess.run(['python','translate_plus.py','data/base/train.json','-l','es',l,'es','-if','text','-of','text','-out',f'data/backtranslation/pipe{i+ofset}.json','-v'])
    subprocess.run(['python','translate_plus.py',f'data/backtranslation/pipe{i+ofset}.json','-l','es',l,'es','-if','headline','-of','headline','-out',f'data/backtranslation/pipe{i+ofset}.json','-v'])
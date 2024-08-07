import subprocess

sizes = [100,250,500,1000,1500,2000]

for size in sizes:
    subprocess.run(['python','add_sel.py',str(size)])
    subprocess.run(['python','train.py','data/ext_data/','models','-v','-exp','ext_extrasel','-run',str(size),'-lr','5e-6','-lr_sch','0.7'])
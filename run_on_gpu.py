import os
import sys

first = True
command = ''
for arg in sys.argv:
    if first:
        first = False
        command = 'optirun /usr/bin/python /home/marius/Documents/Thesis/template_deepnet_aes.py '
    else:
        command += arg + ' '
os.system(command)

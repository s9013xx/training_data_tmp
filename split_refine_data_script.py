import os
import sys

os.system('python split_refine_data.py --device p1000 --convolution')
os.system('python split_refine_data.py --device p1000 --pooling')
os.system('python split_refine_data.py --device p1000 --dense')

os.system('python split_refine_data.py --device p2000 --convolution')
os.system('python split_refine_data.py --device p2000 --pooling')
os.system('python split_refine_data.py --device p2000 --dense')

os.system('python split_refine_data.py --device p4000 --convolution')
os.system('python split_refine_data.py --device p4000 --pooling')
os.system('python split_refine_data.py --device p4000 --dense')

os.system('python split_refine_data.py --device p5000 --convolution')
os.system('python split_refine_data.py --device p5000 --pooling')
os.system('python split_refine_data.py --device p5000 --dense')

os.system('python split_refine_data.py --device 1080ti --convolution')
os.system('python split_refine_data.py --device 1080ti --pooling')
os.system('python split_refine_data.py --device 1080ti --dense')

os.system('python split_refine_data.py --device 2080ti --convolution')
os.system('python split_refine_data.py --device 2080ti --pooling')
os.system('python split_refine_data.py --device 2080ti --dense')
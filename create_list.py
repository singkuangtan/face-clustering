import numpy as np
import os

meta=np.loadtxt('meta.txt')

if os.path.exists("list.txt"):
	os.remove("list.txt") 
out_file=open('list.txt','a')

count=0
for i in range(0,len(meta)):
	out_file.write("{:04d}".format(int(meta[i]))+'/'+"{:06d}".format(count)+'.jpg\n')
	count=count+1
	

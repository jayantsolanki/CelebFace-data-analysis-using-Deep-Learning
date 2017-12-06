#automate.py
import numpy as np
from main import *
for i in np.arange(0.3,0.6,0.1):
	print(i)
	main(i, 32, 64)
	main(i, 64, 128)
	main(i, 128, 256)
	# for j in np.arange(32,49,8):
	# 	for k in np.arange(64,81,8):
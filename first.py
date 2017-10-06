import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as pit
def generate_data():
	np.random.seed(0)
	x,y =  datasets_make_moons(200, noise = 0.20)
	return x,y

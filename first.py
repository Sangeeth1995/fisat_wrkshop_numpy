import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as pit
def generate_data():
	np.random.seed(0)
	X,y =  datasets_make_moons(200, noise = 0.20)
	return X,y
def visualize(X,y,clf):
	plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
	plt.show()

def classify(X, y):
	clf = linear_model.LogisticRegressionCV()
	clf.fit(X,y)
	return clf
def main():
	X,y =  generate_data()
	clf = classify(X, y)
	visualize(X, y , clf)

if __name__ == "__main__":
	main()


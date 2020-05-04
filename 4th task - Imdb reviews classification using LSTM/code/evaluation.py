from keras.datasets import imdb
from keras.models import load_model
from keras.preprocessing import sequence
import sys

#numarul maxim de cuvinte pe care vreti sa le considerati
nr_cuv_diferite = int(sys.argv[1])  # 4000

#dimensiunea maxima a unui review
dim_max = int(sys.argv[2]) # 300

if __name__ == '__main__':

	# Load the dataset
	print(nr_cuv_diferite)
	(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words = nr_cuv_diferite)
	model = load_model("../model.h5")

	X_test = sequence.pad_sequences(X_test, maxlen=dim_max)
	scores = model.evaluate(X_test, Y_test)
	model.summary()

	print('Loss: %.3f' % scores[0])
	print('Accuracy: %.3f' % scores[1])
	
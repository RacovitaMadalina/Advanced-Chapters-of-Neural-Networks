from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence

max_words = 4000
maxlen = 300

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()
train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_words, 32))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

model.fit(train_data, train_labels, batch_size=512, epochs=20, verbose=1, \
          validation_data=(test_data, test_labels))

# Serializing the model
model.save("model.h5")

# Evaluation stage
scores = model.evaluate(test_data, test_labels, batch_size=256, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))
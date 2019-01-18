import preprocessing as prepr
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
import keras.utils

test_size = 0.2

data = prepr.parse_xml_data()

# Split
n_of_articles = len(data)
n_of_test_articles = int(n_of_articles * test_size)
test_articles = data[(n_of_articles - n_of_test_articles):]
train_articles = data[:(n_of_articles - n_of_test_articles)]


train_sentences = []
test_sentences = []
train_tags = []
test_tags = []


for article in train_articles:
    for sentence in article:
        words = []
        tags = []
        for word in article[sentence]['words']:
            words.append(word['word'])
            tags.append(word['pos'])
        train_sentences.append(np.array(words))
        train_tags.append(np.array(tags))
        
        
for article in test_articles:
    for sentence in article:
        words = []
        tags = []
        for word in article[sentence]['words']:
            words.append(word['word'])
            tags.append(word['pos'])
        test_sentences.append(np.array(words))
        test_tags.append(np.array(tags))


print('Training sentences:')
print(train_sentences[0])
print('Length of training sentences: %d' % len(train_sentences))
print('Test sentences:')
print(test_sentences[0])
print('Length of testing sentences: %d' % len(test_sentences))
print('Training tags:')
print(train_tags[0])
print('Length of training tags: %d' % len(train_tags))
print('Testing tags')
print(test_tags[0])
print('Length of testing tags: %d' % len(test_tags))


unique_words, unique_tags = set([]), set([])


for s in train_sentences:
    for w in s:
        unique_words.add(w.lower())

for ts in train_tags:
    for t in ts:
        unique_tags.add(t)

for s in test_sentences:
    for w in s:
        unique_words.add(w.lower)

for ts in test_tags:
    for t in ts:
        unique_tags.add(t)

word2index = {w: i + 2 for i, w in enumerate(list(unique_words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs

tag2index = {t: i + 1 for i, t in enumerate(list(unique_tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])

for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])

print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])

MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)  # Should be 156


train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])


model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH,)))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

model.summary()


categorical_tags_y = keras.utils.to_categorical(train_tags_y, len(tag2index))

model.fit(train_sentences_X, keras.utils.to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=40, validation_split=0.2)
scores = model.evaluate(test_sentences_X, keras.utils.to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")
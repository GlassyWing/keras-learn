from keras.preprocessing.text import Tokenizer

texts = ['That is a good bool!', 'You think that is a good thing.', 'Give you a bool']

tokenizer = Tokenizer(num_words=4)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print(tokenizer.word_counts)
print(sequences)
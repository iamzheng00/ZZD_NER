
# ============TensorFlow2.0教程-Word Embedding===================


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

## 1.载入数据
start = time.time()
vocab_size = 10000
# with tf.device('/gpu:0'):
(train_x, train_y), (test_x, text_y) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(train_x[0])
print(train_y)
print(len(train_x[0]))
print(len(train_x))
print(len(train_y))


word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = {v:k for k, v in word_index.items()}
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(train_x[0]))

# with tf.device('/gpu:0'):
maxlen = 500
train_x = keras.preprocessing.sequence.pad_sequences(train_x,value=word_index['<PAD>'],
                                                    padding='post', maxlen=maxlen)
test_x = keras.preprocessing.sequence.pad_sequences(test_x,value=word_index['<PAD>'],
                                                    padding='post', maxlen=maxlen)

## 2.构建模型

embedding_dim = 100
model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(160, activation='relu'),
    layers.Dense(1, activation='sigmoid')

])
model.summary()


model.compile(optimizer=keras.optimizers.Adam(),
             loss=keras.losses.BinaryCrossentropy(),
             metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=30, batch_size=1024, validation_split=0.1)

print('===========================run time is:', time.time()-start,'seconds!')
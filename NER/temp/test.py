
# ============TensorFlow2.0教程-Word Embedding===================


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import time
import os
import random

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

## 1.载入数据
start = time.time()
vocab_size = 10000
maxlen = 500
batch_size = 512
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

num_batches = len(train_x) // batch_size
x_batches, y_batches = [],[]
for i in range(num_batches):
    x_batches.append(keras.preprocessing.sequence.pad_sequences((train_x[i*batch_size:(i+1)*batch_size]),
                                                                value=word_index['<PAD>'],padding='post', maxlen=maxlen))
    y_batches.append(train_y[i*batch_size:(i+1)*batch_size])


test_x = keras.preprocessing.sequence.pad_sequences(test_x,value=word_index['<PAD>'],
                                                    padding='post', maxlen=maxlen)

## 2.构建模型
embedding_dim = 100


class MyModel(keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.em = layers.Embedding(vocab_size, embedding_dim, input_length=maxlen)
    self.pool = layers.GlobalAveragePooling1D()
    self.d1 = layers.Dense(160, activation='relu')
    self.d2 = layers.Dense(1, activation='softmax')

  def call(self,x,training=None, mask=None):
    x = self.em(x)
    x = self.pool(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()
model.build(input_shape=(None,500))
model.summary()
opt = keras.optimizers.Adam()
acc_f = keras.metrics.Accuracy()
loss_obj = keras.losses.BinaryCrossentropy()
@tf.function
def train(x_batch,y_batch):
    with tf.GradientTape() as tape:
        pred = model(x_batch)
        loss = loss_obj(y_batch,pred)
    grads = tape.gradient(loss,model.trainable_variables)
    acc = acc_f(y_batch,pred)
    opt.apply_gradients(zip(grads,model.trainable_variables))
    return loss,acc

for epoch in range(30):
    for i, x_batch in enumerate(x_batches):
        y_batch = y_batches[i]
        l,a =train(x_batch,y_batch)
        print('epoch:{}\t\tbatch num:{}\t\tloss:{:.4f}\t\tacc:{:.4f}'.format(epoch, i, l, a))

print('===========================run time is:', time.time()-start,'seconds!')
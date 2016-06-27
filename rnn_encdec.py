import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, GRU, Embedding
import numpy as np
import theano
from theano.printing import Print

batch_size = 1
hidden_dim = 2
input_length = 3
output_length = 4
output_dim = 5
voc_size = 6
depth = 3

X_batch = np.zeros(batch_size * input_length, dtype=int).reshape(
    batch_size, input_length)
Y_batch = np.zeros(batch_size * output_length, dtype=int).reshape(
    batch_size, output_length, )

y_i = K.variable(np.ones((batch_size, hidden_dim)))
y = K.variable(np.ones((batch_size, input_length, hidden_dim)))
c = K.variable(np.ones((batch_size, input_length, hidden_dim)))

h = K.variable(np.ones((batch_size, input_length, hidden_dim)))
s = K.variable(np.ones((batch_size, hidden_dim)))


def recurrence(y_i, h):
    h_permute = K.permute_dimensions(h, [0, 2, 1])  # (batch_size, encoding_dim, input_length)
    e = K.l2_normalize(
        K.batch_dot(h_permute, s, axes=1),  # (batch_size, input_length)
        axis=1)  # (batch_size, input_length)

    # eqn 6
    alpha = K.softmax(e)  # (batch_size, input_length)

    # eqn 5
    c = K.batch_dot(h, alpha, axes=1)  # (batch_size, encoding_dim)

    recurrence_result = K.expand_dims(
        K.concatenate([c, y_i], axis=1),
        dim=1)  # (batch_size, 1, 2 * encoding_dim)

    expanded_h = Input(shape=(1, 2 * hidden_dim),
                       name='expanded_h')
    gru = Sequential([
        GRU(output_dim,
            return_sequences=False,
            input_shape=(1, 2 * hidden_dim))
    ])
    model = Model(input=[expanded_h],
                  output=[gru(expanded_h)])  # (batch_size, 1, output_dim)
    return model(recurrence_result)


output, _ = theano.scan(recurrence,
                        sequences=K.permute_dimensions(y, [1, 0, 2]),
                        non_sequences=h)

decode = Lambda(lambda encoded_state: K.permute_dimensions(output, [1, 0, 2]),
                output_shape=(batch_size, output_dim))
decode.build((input_length, hidden_dim))
print(K.eval(decode(h)).shape)

x = Input(shape=(input_length,), dtype='int32', name='x')
embed = Embedding(voc_size, hidden_dim, input_length=input_length)
embed_lookup = embed(x)
gru = GRU(hidden_dim,
          input_shape=(input_length, hidden_dim),
          unroll=True,
          return_sequences=True,
          consume_less='gpu')
encoder = Sequential([embed])
for _ in range(depth):
    encoder.add(gru)
hidden_state = encoder(x)

print(Model(x, hidden_state).predict(X_batch).shape)


def model(train):
    def decoder_unit():
        return GRU(hidden_dim,
                   return_sequences=True,
                   input_shape=(1, hidden_dim))

    reencode = Sequential()
    for _ in range(depth):
        reencode.add(decoder_unit())

    output = decode(hidden_state)
    return Model(input=[x], output=output)


train_model = model(train=True)
test_model = model(train=False)
train_model.compile(loss='binary_crossentropy',
                    optimizer='sgd')
print(train_model.predict([X_batch]))  # , Y_batch]))

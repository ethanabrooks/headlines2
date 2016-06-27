import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, GRU
import numpy as np
import theano
from theano.printing import Print

batch_size = 2
encoding_dim = 3
input_length = 4
output_length = 5
output_dim = 6

y_i = K.variable(np.ones((batch_size, encoding_dim)))
c = K.variable(np.ones((batch_size, input_length, encoding_dim)))

h = K.variable(np.ones((batch_size, input_length, encoding_dim)))
s = K.variable(np.ones((batch_size, encoding_dim)))


def recurrence(y_i, h):
    h_permute = K.permute_dimensions(h, [0, 2, 1])  # (batch_size, encoding_dim, input_length)
    e = K.l2_normalize(
        K.batch_dot(h_permute, s, axes=1),  # (batch_size, input_length)
        axis=1)  # (batch_size, input_length)

    # eqn 6
    alpha = K.softmax(e)  # (batch_size, input_length)

    # eqn 5
    c = K.batch_dot(h, alpha, axes=1)  # (batch_size, encoding_dim)

    dim_ = [c, y_i]
    return dim_
    # recurrence_result = K.concatenate(dim_,
    #                                   axis=1),  # (batch_size, 1, 2 * encoding_dim)
    # expanded_h = Input(shape=(1, 2 * encoding_dim),
    #                    name='expanded_h')
    # gru = Sequential([
    #     GRU(output_dim,
    #         return_sequences=False,
    #         input_shape=(1, 2 * encoding_dim))
    # ])
    # model = Model(input=[expanded_h],
    #               output=[gru(expanded_h)])  # (batch_size, 1, output_dim)
    # return model(recurrence_result)

# output, _ = theano.scan(recurrence,
#                         sequences=y,
#                         non_sequences=s)
for result in recurrence(y_i, h):
    print('-' * 10)
    print(K.eval(result))

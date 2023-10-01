# model.py
import tensorflow as tf

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def compile_model(model):
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

def train_model(model, dataset, epochs):
    for epoch in range(epochs):
        model.fit(dataset, epochs=1, verbose=1)
        model.reset_states()

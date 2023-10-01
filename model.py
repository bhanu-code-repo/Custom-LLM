# model.py
import tensorflow as tf

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Builds a model using the given parameters.

    Parameters:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding.
        rnn_units (int): The number of units in the LSTM layer.
        batch_size (int): The batch size.

    Returns:
        tf.keras.Sequential: The built model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def compile_model(model):
    """
    Compiles the given model with the specified optimizer and loss function.

    Parameters:
        model (tf.keras.Model): The model to be compiled.

    Returns:
        None
    """
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

def train_model(model, dataset, epochs):
    """
    Train the model using the given dataset for a specified number of epochs.

    Parameters:
        model (object): The model to be trained.
        dataset (object): The dataset to be used for training.
        epochs (int): The number of epochs for training the model.

    Returns:
        None
    """
    for epoch in range(epochs):
        model.fit(dataset, epochs=1, verbose=1)
        model.reset_states()

# data_utils.py
import tensorflow as tf

def load_text_data(file_path):
    """
    Load text data from a file.

    Parameters:
        file_path (str): The path to the file that contains the text data.

    Returns:
        str: The text data loaded from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def create_dataset(text, seq_length, batch_size):
    """
    Creates a dataset for training a language model.

    Args:
        text (str): The input text to create the dataset from.
        seq_length (int): The length of each input sequence.
        batch_size (int): The size of each training batch.

    Returns:
        dataset (tf.data.Dataset): The created dataset.
        vocab (list): The vocabulary list containing all unique characters in the text.
        char_to_int (dict): A dictionary mapping each character to its corresponding integer index.
        int_to_char (dict): A dictionary mapping each integer index to its corresponding character.
    """
    vocab = sorted(set(text))
    char_to_int = {char: i for i, char in enumerate(vocab)}
    int_to_char = {i: char for i, char in enumerate(vocab)}

    text_as_int = [char_to_int[char] for char in text]

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)

    return dataset, vocab, char_to_int, int_to_char

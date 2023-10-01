# text_generation.py
import tensorflow as tf

def generate_text(model, start_string, char_to_int, int_to_char, num_generate=1000, temperature=1.0):
    """
    Generates text using a given model.

    Parameters:
    - model: The model used for text generation.
    - start_string: The initial string used to start text generation.
    - char_to_int: A dictionary mapping characters to integer values.
    - int_to_char: A dictionary mapping integer values to characters.
    - num_generate: The number of characters to generate.
    - temperature: The randomness of the generated text.

    Returns:
    - The generated text starting from the given start string.
    """
    input_eval = [char_to_int[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    
    generated_text = []

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        generated_text.append(int_to_char[predicted_id])

    return start_string + ''.join(generated_text)

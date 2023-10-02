# app.py
from flask import Flask, render_template, request
import tensorflow as tf
import data_utils
import text_generation

app = Flask(__name__)

# Load the saved model
lm_model = tf.keras.models.load_model('my_language_model')

# Load data and vocabulary
text_data = data_utils.load_text_data('text_data.txt')
_, _, char_to_int, int_to_char = data_utils.create_dataset(text_data, SEQ_LENGTH, BATCH_SIZE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    seed_text = request.form['seed_text']
    generated_text = text_generation.generate_text(lm_model, seed_text, char_to_int, int_to_char)
    return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)

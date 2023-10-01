# main.py
import data_utils
import model
import text_generation

# Hyperparameters
SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 50

# Load and preprocess the data
text_data = data_utils.load_text_data('text_data.txt')
dataset, vocab, char_to_int, int_to_char = data_utils.create_dataset(text_data, SEQ_LENGTH, BATCH_SIZE)

# Build and compile the model
lm_model = model.build_model(len(vocab), EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.compile_model(lm_model)

# Train the model
model.train_model(lm_model, dataset, EPOCHS)

# Generate text
seed_text = "Once upon a time"
generated_text = text_generation.generate_text(lm_model, seed_text, char_to_int, int_to_char)
print(generated_text)

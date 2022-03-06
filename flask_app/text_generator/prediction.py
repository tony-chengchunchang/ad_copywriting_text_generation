from flask import Blueprint, request, render_template
# import tensorflow as tf
from text_generator.libs import TextGenModel, TextGenerator
import pickle

TOKENIZER_FILE_PATH = 'text_generator/tokenizer.pkl'
MODEL_WEIGHT_PATH = 'text_generator/test_model'
VOCAB_SIZE = 2648
EMBEDDING_DIM = 256
RNN_UNITS = 1024

prediction_route = Blueprint('prediction', __name__)

with open(TOKENIZER_FILE_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

model = TextGenModel(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS)
model.load_weights(MODEL_WEIGHT_PATH)

def predict(start_text, text_length, temperature=0.8, sample_count=1):
    text_gen = TextGenerator(model, tokenizer)
    results = []
    for i in range(int(sample_count)):
        result = text_gen.generate_texts(start_text, int(text_length), float(temperature))
        results.append(result)

    return results

@prediction_route.route('/text_gen', methods=['GET', 'POST'])
def make_prediction():
    results = []
    params = {'sample_count': 1, 'temperature': 0.8}
    if request.method == 'POST':
        params = request.form
        results = predict(**params)

    return render_template('text_gen.html', results=results, **params)

import tensorflow as tf

class TextGenModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)

        x, states = self.gru(x, initial_state=states, training=training)
        outputs = self.dense(x, training=training)

        if return_state:
            return outputs, states
        else:
            return outputs

class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def to_text(self, tokens):
        '''
        transfer tokens back to texts
        '''
        text = ''
        for t in tokens:
            word = self.tokenizer.index_word[t]
            text += word

        return text

    def generate_one_char(self, inputs, states=None, temperature=0.5):
        # sequence can be in any length, does not have to be the same as the training data
        sequence = self.tokenizer.texts_to_sequences([inputs])[0]

        # add batch dim
        inputs_seq = tf.expand_dims(sequence, axis=0)

        # call the model
        # difference btw call() & predict():
        # predict() handles larger inputs and can do multi batches
        # call() handles single batch
        pred_logits, states = self.model(inputs_seq, states=states, return_state=True)

        # each time a char passed in, rnn will gen a pred and pass the pred and state into the model for next pred
        # thus there will be serveral pred_logits according to how many inputs(chars) passed in

        # get the last pred cause that is the only result that we really want to predict
        # eg. 'ab' => 'abc', the 'c' is the only thing we care about
        last_char_pred_logits = pred_logits[:,-1,:]

        # devide the last_char_pred_logits by temperature(certain num)
        # the greater temperature means greater diversity
        last_char_pred_logits /= temperature
        pred_id = tf.random.categorical(last_char_pred_logits, num_samples=1)
        pred_id = tf.squeeze(pred_id, axis=1)
        pred_char = self.to_text(pred_id.numpy())

        return pred_char, states

    def generate_texts(self, start_text, text_length, temperature=0.5):
        states = None
        outputs_seq = [start_text]
        next_text = start_text

        for i in range(text_length):
            next_text, states = self.generate_one_char(next_text, states=states, temperature=temperature)
            outputs_seq.append(next_text)

        result = ''.join(outputs_seq)

        return result

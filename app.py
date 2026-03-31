from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

with open("en_tokenizer.pickle", "rb") as f:
    en_tokenizer = pickle.load(f)

with open("fr_tokenizer.pickle", "rb") as f:
    fr_tokenizer = pickle.load(f)

MAX_LEN = 25
EN_VOCAB_SIZE = len(en_tokenizer.word_index) + 1
FR_VOCAB_SIZE = len(fr_tokenizer.word_index) + 1
EMBED_DIM = 64
LATENT_DIM = 128

encoder_inputs = Input(shape=(MAX_LEN,))
enc_emb_layer = Embedding(EN_VOCAB_SIZE, EMBED_DIM, mask_zero=True)
enc_emb = enc_emb_layer(encoder_inputs)

encoder_lstm = LSTM(LATENT_DIM, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(MAX_LEN - 1,))
dec_emb_layer = Embedding(FR_VOCAB_SIZE, EMBED_DIM, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(FR_VOCAB_SIZE, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.load_weights("model.weights.h5")

encoder_model = Model(encoder_inputs, encoder_states)

decoder_input_single = Input(shape=(1,))
h_in = Input(shape=(LATENT_DIM,))
c_in = Input(shape=(LATENT_DIM,))

dec_emb_single = dec_emb_layer(decoder_input_single)
y, h, c = decoder_lstm(dec_emb_single, initial_state=[h_in, c_in])
y = decoder_dense(y)

decoder_model = Model([decoder_input_single, h_in, c_in], [y, h, c])

start_id = fr_tokenizer.word_index["<start>"]
end_id = fr_tokenizer.word_index["<end>"]

def translate(sentence, max_len=30):
    seq = en_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    h, c = encoder_model.predict(seq, verbose=0)
    target_seq = np.array([[start_id]])

    output_words = []

    for _ in range(max_len):
        y, h, c = decoder_model.predict([target_seq, h, c], verbose=0)
        next_id = int(np.argmax(y[0, 0, :]))

        if next_id == end_id:
            break

        word = fr_tokenizer.index_word.get(next_id, "")
        output_words.append(word)
        target_seq = np.array([[next_id]])

    return " ".join(output_words)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("text", "").strip()
        if user_input:
            result = translate(user_input)

    return render_template("index.html", result=result, user_input=user_input)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
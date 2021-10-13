import tensorflow as tf

def preprocess_sentence(w):
    w = '#' + w + '@'
    return w.split()

def evaluate(sentence, max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder):
    sentence = preprocess_sentence(sentence)[0]

    inputs = [inp_lang.word2idx[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['@']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if targ_lang.idx2word[predicted_id] == '@':
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence
      
def transliterate(sentence, max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder):
    result, sentence = evaluate(sentence, max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder)
    return ''.join(result.split(' ')).replace('#','').replace('@','')

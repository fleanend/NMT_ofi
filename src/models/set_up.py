from src.data import make_dataset
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.bahdanau_attention import BahdanauAttention
import tensorflow as tf
import os


def set_up_model(data_path = 'data/set_up_data.txt', checkpoint_dir_path = 'models'):
    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([targ_lang.word2idx['@']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
        
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # Prepare dataset with all the proper characters and properties to set all the variables as they were during training
    data = []

    with open(data_path, encoding='utf-8') as f:
            for line in f.readlines():
                data.append(line)

    data = make_dataset.create_dataset(data)

    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = make_dataset.load_dataset(data)

    # Set up model as if we were to train it, to define graph and ease the checkpoint loading step
    BUFFER_SIZE = len(input_tensor)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_tensor)//BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(inp_lang.word2idx)+1
    vocab_tar_size = len(targ_lang.word2idx)+1
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    example_input_batch, example_target_batch = next(iter(dataset))

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                          sample_hidden, sample_output)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
        
    checkpoint_prefix = os.path.join(checkpoint_dir_path, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
                                                    
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        break

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir_path))
    
    # At this point the model is set up for prediction and all auxiliary variables and data structures are working properly
    
    return max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder
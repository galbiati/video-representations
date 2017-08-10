import tensorflow as tf
import numpy as np

def get_l2_term(exclude_names=['lstm']):
    l2_vars = [v for v in tf.trainable_variables() if 'bias' not in v.name]
    for name in exclude_names:
        l2_vars = [v for v in l2_vars if name not in v.name]
    return tf.add_n([tf.nn.l2_loss(v) for v in l2_vars])


def train(model, num_epochs=10, batchsize=4, savefile=None):
    """
    Trains the convolutional-LSTM autoencoder for num_epochs, optionally
    saving session to savefile

    Args:
    ------
    :model is an instance of Model() from models/model.py
    :num_epochs is the total number of passes over the training data
    :batchsize is the size of minibatches during training
    :savefile is the NAME (not filepath!) into which to save the session

    Outputs:
    ------
    :losses is a list of per-batch MSE losses during training
    """
    assert model.batchsize == batchsize
    ## LSTM-Encoder Training Graph ##
    training_inputs, training_targets = load.inputs('training', batchsize, training_epochs)

    encoded, transitioned, decoded = model.build(training_inputs)    # discard decoder here
    loss = tf.reduce_mean(tf.pow(decoded - training_targets, 2))

    optimizer = tf.train.AdamOptimizer()
    trainable_vars = tf.trainable_variables()
    clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars), 1)    # clip those uglies
    train_step = optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))

    ## LSTM-Encoder Validation Graph ##

    validation_inputs, validation_targets = load.inputs('validation', batchsize, 1)

    encoded_validation, transitioned_validation, decoded_validation = model.build(validation_inputs, reuse=True)
    targeted_validation = model.build_target_encoder(validation_targets, reuse=True)
    validation_loss = tf.reduce_mean(tf.pow(decoded_validation - validation_targets, 2))

    saver = tf.train.Saver()
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    coord = tf.train.Coordinator()

    with tf.Session() as sesh:
        sesh.run([init_global, init_local])
        threads = tf.train.start_queue_runners(sess=sesh, coord=coord)

        # initialize lists for tracking

        decoder_losses = []
        decoder_validation_losses = []

        predictions = []
        encodings = []
        transitions = []
        validation_predictions = []
        validation_transitions = []
        validation_encodings = []
        recovery = []
        validation_recovery = []

        # first, encoder training
        try:
            step = 0

            while not coord.should_stop():
                _, loss_value, enc, trans, pred, input_recover = sesh.run(
                    [train_step, loss, encoded, transitioned, decoded, training_targets]
                )

                decoder_losses.append(loss_value)

                if step % 250 == 0:
                    print('Step {}, loss: {:.2f}'.format(step, loss_value))
                    encodings.append(enc)
                    transitions.append(trans)
                    predictions.append(pred)
                    recovery.append(input_recover)

                step += 1

        except tf.errors.OutOfRangeError:
            print('Encoder trained: {:.2f}'.format(loss_value))

        # second, encoder validation
        try:
            step = 0

            while not coord.should_stop():
                _, loss_value, enc, trans, pred, input_recover = sesh.run(
                    [validation_loss, encoded_validation, transitioned_validation,
                     decoded_validation, validation_targets]
                )
                decoder_validation_losses.append(loss_value)

                if step % 100 == 0:
                    print(step, loss_value)
                    validation_encodings.append(enc)
                    validation_transitions.append(trans)
                    validation_predictions.append(pred)
                    validation_recovery.append(input_recover)

                step += 1

        except tf.errors.OutOfRangeError:
            print('Encoder validated: {:.2f}'.format(loss_value))

        finally:
            coord.request_stop()

        coord.join(threads)

        if savefile is not None:
            saver.save(sesh, savefile)

            # also cache outputs for another day
            np.savez_compressed('training_encodings.npz', *encodings)
            np.savez_compressed('training_transitons.npz', *transitions)
            np.savez_compressed('training_predictions.npz', *predictions)
            np.savez_compressed('training_recovery.npz', *recovery)

    return losses

def iterate_minibatches(inputs, targets, batchsize=32, shuffle=False):
    """Utility func for shuffling minibatches with placeholder inputs"""
    assert len(inputs) == len(targets)
    if shuffle:
        idxs = np.arange(len(inputs))
        np.random.shuffle(idxs)

    for idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = idxs[idx:idx + batchsize]
        else:
            excerpt = slice(idx, idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

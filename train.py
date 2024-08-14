from preprocess import *
import tensorflow as tf
import math
import keras

def create_sequences(dataset, seq_length, vocab_size):
    seq_length += 1
    
    def scale(inputs):
        return inputs/[vocab_size, 1, 1]

    def process_batch(batch):
        features = batch[:-1]
        labels = batch[-1]
        dic = {'pitch': labels[0], 'step': labels[1], 'duration': labels[2]}
        return scale(features), dic
    
    return dataset.batch(seq_length, drop_remainder = True).map(process_batch, num_parallel_calls = tf.data.AUTOTUNE)
        
def create_model(seq_length):

    @keras.saving.register_keras_serializable(package="my_package", name="mse_with_positive_pressure")
    def mse_with_positive_pressure(y:tf.Tensor, pred: tf.Tensor): #from https://www.kaggle.com/code/arielcheng218/generating-lofi-music-with-rnns
        mse = (y - pred) ** 2
        if pred < 0:
            pressure = 100 * -pred
        else:
            pressure = 0
        return tf.reduce_mean(mse + pressure)
    
    input_shape = (seq_length, 3)
    learning_rate = .005

    inputs = tf.keras.Input(input_shape)
    output = tf.keras.layers.LSTM(128)(inputs)

    pitch_output = tf.keras.layers.Dense(128, name = 'pitch')(output)
    step_output = tf.keras.layers.Dense(1, name = 'step')(output)
    duration_output = tf.keras.layers.Dense(1, name = 'duration')(output)

    model = tf.keras.Model(inputs, [pitch_output, step_output, duration_output])

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    model.compile(loss = loss, optimizer = optimizer)

    return model
    


def main():
    checkpoint_path = "working/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    train_dataset = tf.data.Dataset.from_tensor_slices(np.loadtxt('notes_melody.txt', dtype = float))
    seq_length = 16
    vocab_size = 128
    seq_dataset = create_sequences(train_dataset, seq_length, vocab_size)
    batch_size = 20
    train_dataset = (seq_dataset.batch(batch_size, drop_remainder = True)).cache().prefetch(tf.data.experimental.AUTOTUNE)

    model = create_model(seq_length)

    model.save_weights(checkpoint_path.format(epoch = 0))
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = checkpoint_path,
            save_weights_only = True,
            save_freq= 'epoch',  
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor = 'loss',
            patience = 5,
            verbose = 1,
            restore_best_weights = True
        )
    ]

    epochs = 30

    try:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print("Data Loaded\n")
    except:
        print("Not Loaded\n")
    
    model.save_weights(checkpoint_path.format(epoch=0))

    history = model.fit(train_dataset, epochs = epochs, callbacks = callbacks)

    model.save("working/best_model/trained_model.keras")

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('epoch_loss.png')

if __name__=='__main__':
    main()
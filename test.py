from train import *
import random

def predict_next_note(notes, model, temp, last_note):
    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs)
    while True:
        predictions = model.predict(inputs)
        pitch_logits = predictions['pitch']
        pitch_logits /= temp
        pitch = tf.random.categorical(pitch_logits, num_samples = 1)
        pitch = tf.squeeze(pitch, axis = -1)
        pitch = pitch.numpy()
        diff = (abs(last_note - pitch[0]))
        rand = random.random()
        if diff % 4 == 0 or diff % 7 == 0 and diff != 0:
            break
        elif rand < .3:
            break
    step = predictions['step']
    duration = predictions['duration']
    duration = tf.squeeze(duration, axis = -1)
    step = tf.squeeze(step, axis = -1)
    if step < 0:
        step = 0
    if duration < 0:
        duration = 0

    return int(pitch), float(step), float(duration)

def notes_to_midi(notes, pd_file, instrument_name, velocity):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program = pretty_midi.instrument_name_to_program(instrument_name))
    for i, note in notes.iterrows():
        note = pretty_midi.Note(
            velocity,
            pitch = int(note['pitch']),
            start = note['start'],
            end = note['end']
        )
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    midi.write(pd_file)
    return midi

def main():
    key_order = ['pitch', 'step', 'duration']
    seq_length = 16
    vocab_size = 128

    num_predictions =   120
    sample = pretty_midi.PrettyMIDI(sample_file)
    all_midi_notes = midi_file_to_notes(sample)
    last_note = all_midi_notes.iloc[seq_length]['pitch']
    sample_notes = np.stack([all_midi_notes[key] for key in key_order], axis = 1)

    input_notes = (sample_notes[:seq_length]/np.array([vocab_size, 1, 1]))
    song = []
    prev_start = 0

    model = tf.keras.models.load_model('working/best_model/trained_model.keras', compile=False)

    print(model.summary())

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, 1.25, last_note)
        last_note = pitch
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)  #from https://www.kaggle.com/code/arielcheng218/generating-lofi-music-with-rnns
        song.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis = 0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis = 0)
        prev_start = start

    song = pd.DataFrame(song, columns = (*key_order, 'start', 'end'))
    start = 0
    for i, row in song.iterrows():
        nstart = row['duration'] + start
        row['start'] = start
        row['end'] = nstart
        start = nstart
        song.iloc[i] = row
    print(song)


    out_file = 'output.mid'
    instrument = sample.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    notes_to_midi(generated_notes, out_file, instrument_name, 10)
    

if __name__ == "__main__":
    main()
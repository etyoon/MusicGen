import pretty_midi
import os
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "melody/"
sample_file =  "melody/ashover10.mid"
 
def midi_file_to_notes(midi_file):
    notes = midi_file.instruments[0].notes
    prev_start = notes[0].start
    dic = {'pitch': [], 'step': [], 'duration': []}

    for note in notes:
        dic['pitch'].append(note.pitch)
        dic['step'].append(note.start - prev_start)
        dic['duration'].append(note.end - note.start)

    df = pd.DataFrame(dic)
    df_sorted = df.sort_values(by='step')

    return df_sorted

def get_notes():
    notes = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        midi = pretty_midi.PrettyMIDI(path)
        notes.append(midi_file_to_notes(midi))
    notes = pd.concat(notes)
    print(len(notes))
    note_count = notes['pitch'].value_counts().reset_index()
    note_count.columns = ['pitch', 'count']
    note_count.plot.bar(x = 'pitch', y = 'count', rot = 0)
    plt.savefig('occurences.png')
    valid_notes = note_count[note_count['count']>=100]['pitch']
    filtered_df = notes[notes['pitch'].isin(valid_notes)]
    print(len(filtered_df))
    np.savetxt('notes_melody.txt', np.array(filtered_df), fmt='%10.5f')

if __name__ == '__main__':
    get_notes()
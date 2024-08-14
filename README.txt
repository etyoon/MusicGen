preprocessing.py - python script to turn midi files into a pandas dataframe denoting pitch, step, and duration. 
It then filters out any notes that are seen less than 100 times in the dataframe and saves the dateframe to notes_melody.txt.
To run simpy write python3 preprocessing.py

train.py - python script to build and train the recurrent neural network. After training has complete, the model will be saved under the 
working directory called best_model.keras so it can be loaded as a model later.

test.py - python script to generate 120 notes tranform them into a midi file. Loads the saved model from the working directory.The prediction 
tries to append notes that are a major fifth or third away from the previous note. Will create an output.mid file with the generated melody in it

before running script activate the virtual environment in the env directory
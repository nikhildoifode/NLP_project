# Using a KG-Copy Network for Non-Goal Oriented Dialogues

## Requirements

- python 3.6
- pytorch 1.2.0
- Run ```conda create -n nlp_proj python=3.6```
- Run ```conda activate nlp_proj```
- Run ```pip install -r requirements.txt```

**NOTE: The following pre-processing step is not required if you just want to train the system on our processed data (since all the required pre-processed data are included in the project directory).

## Pre-processing

### Building Knowledge Graph

Running the following code will download information from wikipedia and will create a Knowledge Graphs for clubs and national teams respectively. Names of the selected clubs and national teams are currently hard-coded into the 'build_KG_clubs.py','build_KG_national_teams.py' files:

```
python kg_build/build_KG_clubs.py
python kg_build/build_KG_national_teams.py
```

### Building vocabulary

In order to build a vocabulary for the system, run the following command. Running the commands will create vocabulary for the system for the given KGs (which we have already built in the previous step) :

```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
mv wiki.en.vec vocab/
python create_vocab_kb.py
```

Running the commands will generate 'glove300.npy','vocab.npy','w2i.npy' files inside 'vocab/' directory

### Generating train/test/dev data from AMT data (soccer conversations)

To create and preprocess train-test-dev data, run the following command (Train, test, validation data are already pre-processed and generated inside \preproc_files  directory).
No need to preprocess again if you just want to train/test the model.

```
python -m spacy download en_core_web_sm
python -m spacy download en
python -m spacy download en_core_web_lg

python preprocess_kb_2.py --data_dir soccer_conversations/
python utils/generate_entities_soccer.py
```

## Train & Test

Pre-processing is not required if you just want to train/test the model at this point. To train the system run the following command:

For Soccer Domain:

```
python -u ./train_kg_copy.py --batch_size 32 --hidden_size 128 --teacher_forcing 12 --resp_len 10 --lr 0.001 --num_layer 1 --gpu 1 --epochs 150 --data_dir preproc_files/soccer/
```

In each epochs the best trained model so far will be saved inside '/models' directory with a file name 'Sentient_model2.bin'. The saved model can later be used for testing purpose on new data.
After completing the training the command will also generate a file 'test_predicted_kg_attn.csv' where we can check predicted output along with given input test data.

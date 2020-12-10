# Designing a non-goal oriented Question Answering System for Soccer

## Requirements

This project is implemented in python 3.6 and pytorch 1.2.0. Follow these steps to setup your environment:

- [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
- Create a Conda environment with Python 3.6: ```conda create -n nlp_proj python=3.6```
- Activate the Conda environment: ```conda activate nlp_proj```
- Install the requirements: ```pip install -r requirements.txt```

> NOTE: The following pre-processing step is not required if you just want to train/test the system on our processed data (since all the required pre-processed data are included in the project directory).

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

> NOTE: The training step is not required if you just want to test the system on proposed model (since the required model is already included in the project directory). You can just run the test command in this case.

To train the system run the following command:

For training with GPU:

```
python -u ./train_kg_copy.py --batch_size 32 --hidden_size 128 --teacher_forcing 12 --resp_len 10 --lr 0.001 --num_layer 1 --gpu 1 --epochs 150 --data_dir preproc_files/soccer/
```

For training without GPU:

```
python -u ./train_kg_copy.py --batch_size 32 --hidden_size 128 --teacher_forcing 12 --resp_len 10 --lr 0.001 --num_layer 1 --epochs 150 --data_dir preproc_files/soccer/
```

In each epoch, the best trained model so far will be saved inside '/models' directory with a file name 'Sentient_model2.bin'. The saved model can later be used for testing purpose on new data.

To test the system run the following command:

For testing with GPU:

```
python -u ./test_kg_copy.py --batch_size 32 --hidden_size 128 --teacher_forcing 12 --resp_len 10 --lr 0.001 --num_layer 1 --gpu 1 --epochs 150 --data_dir preproc_files/soccer/
```

For testing without GPU:

```
python -u ./test_kg_copy.py --batch_size 32 --hidden_size 128 --teacher_forcing 12 --resp_len 10 --lr 0.001 --num_layer 1 --epochs 150 --data_dir preproc_files/soccer/
```

After running the test command, a file 'test_predicted_kg_attn.csv' will be generated where we can check predicted output along with given input test data.

## Original Source

### Dataset

Thanks to the authors for making the dataset public - https://github.com/SmartDataAnalytics/KG-Copy_Network/tree/master/soccer_conversations

### Baseline Source Code

- Key value memory networks - https://github.com/sunnysai12345/KVMemnn
- KG-Copy_network - https://github.com/SmartDataAnalytics/KG-Copy_Network
- GUpdater - https://github.com/esddse/GUpdater

### New Additions

- Seperated training and testing code
- Added new questions for testing purpose
- Updated Sentient Attention class to increase the similarity score between the senetence embedding of question and knowledge graph.
- Updated Sentient Attention class by adding layers for sentinel function
- Scraping club info for KG from Wikipedia in correct format

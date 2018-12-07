# PreTraining
For PreTraining :
python pretrain.py

## config.py:
In the config.py specify the a network
Set the training type flag to Pretrain
STARTING-EPOCH and EPOCH refer to where to start training - if starting from scratch set Stating =0

## network.py:
Define your neytwork here

## initialization.py:
initialize here

## imdb-extract.py:
Logic to load the imdb datasets

## imdb-preprocess.py
Logic to Pre-Process the loaded imdb dataframes and generate index word mappings



# Finetune
For finetuning a pretrained n/w
Without Embedding: python finetune.py / With Embedding: python finetune_embed.py
The networks are defined in network.py but the initialization is done in finetune.py orr finetune_embed.py
Specify the network to start from in config.py

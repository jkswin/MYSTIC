"""
Train a NN with X parameters on 10,000 sentences.

"""

import json
from pos_neural import SeqNeuralNet
from bpe import BPE
import numpy as np
from utils import rmse, mse, mae
import matplotlib.pyplot as plt
import random

np.random.seed(42)
random.seed(42)

encoder = BPE.from_pretrained()

with open("projects/text_adventure/data/training_data.json", "r") as f:
    examples = [ex for ex in json.load(f) if "ã" not in ex["review"]]
    random.shuffle(examples)
    train_examples = [example["review"] for example in examples]
    train_labels = [example["sentiment_score"] for example in examples]

max_len=20
vocab_size = len(encoder.id2tok.keys())
embedding_size = 50

X_train = np.array([encoder.tokenize_to_ids(example, max_len=max_len) for example in train_examples])

y_train = np.array(train_labels)
y_train = y_train.reshape(y_train.shape[0], 1)


net = SeqNeuralNet(
        layer_sizes= [max_len, 512, 512, 512, 512, 512, 1], # first hidden layer takes output of embedding layer
        activations= ["elu", "elu", "elu", "elu", "elu", "linear"],
        loss = "mse",
        seed=42,
        lr=0.00005,
        norm=True,
        add_positional=True, # whether or not to apply positional encoding
        max_len=max_len, # length of each sentence in tokens
        embedding_dim=embedding_size, # desired vector size for word embeddings
        vocab_size=vocab_size, # number of unique tokens in vocab
        clip=[-1, 1] # gradient clipping to prevent vanishing/exploding
        )


losses = net.train(X=X_train, 
                   y=y_train, 
                   epochs=500, 
                   batch_size=128, 
                  )

path="projects/text_adventure/models/seqnet_linear_enhanceddata.pkl"

y_hat = net.predict(X_train)


RMSE = {"rmse": rmse(y_pred=y_hat, y_true=y_train)}
MSE = {"mse": mse(y_pred=y_hat, y_true=y_train)}
MAE = {"mae": mae(y_pred=y_hat, y_true=y_train)}
[net.evaluation.update(ev) for ev in [RMSE, MSE, MAE]]
net.save_pretrained(path)

plt.plot(range(len(losses)), losses)
plt.xlabel("Epoch")
plt.ylabel(net.loss.__name__.title() + " Loss")
plt.show()


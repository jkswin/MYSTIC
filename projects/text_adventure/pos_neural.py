"""
A feed forward neural network from scratch with some additional whistles and bells:
    - An Embedding Layer that learns jointly with the net during training
    - Positional Encoding
    - He Initialisation
    - Layer Normalisation

    
Vanilla NNs assume independence between features and so aren't suited to processing sequential data like sentences. 
Recurrent architectures are slow to train, and given that this is not optimized would be even slower.
The Transformer adds positional encodings to help circumvent the need for recurrence. 

Out of interest, I build this mini architecture that applies positional encodings to a feed forward net.
I don't anticipate great performance given that it lacks attention blocks.

Also imports implementation of Byte-Pair-Encoding.

"""

import numpy as np
from typing import Any, List
import pickle
from datetime import datetime

from utils import *



class PositionalEncoding:
    """
    Fits positional encoding to a given matrix. Inspired by MachineLearningMastery
    """

    def __init__(self, seqlen, output_dim, n=10_000) -> None:
        """
        Initialise the PositionalEncoding object.
        """
        self.seqlen = seqlen
        self.output_dim = output_dim
        self.n = n

        self.P = np.zeros((seqlen, output_dim), dtype=np.float64)
        self._encode()


    def _encode(self):
        """
        Calculate the encoding matrix for the specified dimensions.
        """
        for k in range(self.seqlen):
            for i in np.arange(int(self.output_dim/2)):
                denominator = np.power(self.n, 2*i/self.output_dim, dtype=np.float64)
                even = np.sin(k/denominator)
                odd = np.cos(k/denominator)
                self.P[k, 2*i] = even
                self.P[k, (2*i)+1] = odd

        return self.P
    

    def encode(self, sentences):
        """
        Expects 3 dimensions. n_sentences, n_tokens, n_dimensions of word vec
        Adds encodings element wise as in Vaswani et al. 2017
        """
        return np.sum((sentences, self.P[np.newaxis,:,:]))
    

class FFLayer:
    """
    A fully connected feed forward layer.
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str, lr: float, seed=False) -> None:

        """
        Initializes the FFLayer object.
        """

        self.activation_functions = {
        "relu": [relu, relu_prime],
        "elu": [elu, elu_prime],
        "softmax": [softmax, softmax_prime],
        "linear": [linear, linear_prime],
        "sigmoid": [sigmoid, sigmoid_prime],
        }

        if seed:
            np.random.seed(seed)

        self.lr = lr
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(1, output_dim)
        self.act, self.act_prime = self.activation_functions.get(activation)
        self._weight_init(activation)


    def __str__(self) -> str:

        """
        Returns a string representation of the FFLayer object.

        """

        return f"DenseFeedForward\nInputs: {self.W.shape[1]}\nOutputs: {self.W.shape[0]}\nActivation: {self.act.__name__}\n"
    

    def _weight_init(self, activation:str):
        """
        Apply He Initialisation to relu-based activations.
        """
        if activation.endswith("elu"):
            self.W = relu_init(self.W)    


    def forward(self, inputs, norm=False):
        """
        Computes the forward pass for the layer with optional layer normalization.
        """
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.W.T) + self.b

        if norm:
            x = self.outputs
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.mean(((x - mean) ** 2), axis=-1, keepdims=True)
            std = np.sqrt(var + 1e-8)
            self.outputs = (x - mean) / std

        self.activations = self.act(self.outputs)

        return self.activations


    def backprop(self, dA, clip=[-1,1]):
        """
        Computes the backward pass for the layer. Clips gradients to between -1 and 1.
        """
        dZ = self.act_prime(self.outputs) * dA 
        dW = 1/dZ.shape[0] * np.dot(dZ.T, self.inputs)
        db = 1/dZ.shape[0] * np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.W)

        dW = np.clip(dW, clip[0], clip[1])
        db = np.clip(db, clip[0], clip[1])
        dA_prev = np.clip(dA_prev, clip[0], clip[1])

        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

        return dA_prev
    

class EmbeddingLayer():
    """
    A Trainable Embedding Layer
    """

    def __init__(self, vocab_size: int, embedding_dim: int, lr: float, seed=42, positional_encodings=False, max_len=None):
        """
        Initialise an EmbeddingLayer
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim)
        self.lr = lr
        self.positional_encodings = positional_encodings
        self.max_len = max_len
        self.pos_encoder = None

        if positional_encodings:
            if self.max_len is None:
                raise ValueError("Cannot apply positional encoding without specifying length of input sequences.")
            self.init_positional()

    def __str__(self) -> str:
        """
        Returns a string representation of the EmbeddingLayer object.

        """

        return f"Embedding\nEmbedding Dimension: {self.embedding_dim}\nOutputs: {self.max_len * self.embedding_dim}\n"

    def forward(self, inputs):
        """
        Converts TokenIDs to Word Embeddings, applies positional encoding and flattens the output.
        """
        self.inputs = inputs
        self.embedded_inputs = self.embedding_matrix[inputs]

        #apply positional encoding
        if self.pos_encoder is not None:
            self.embedded_inputs = self.pos_encoder.encode(self.embedded_inputs)

        return self.flatten(self.embedded_inputs)


    def backprop(self, dA, clip=[-1,1]):
        """
        Updates the Embeddings using the gradients of the previous layer. 
        The resulting embeddings are task specific, but can in theory be used elsewhere.
        """
        dE = np.zeros_like(self.embedding_matrix)
        np.add.at(dE, self.inputs, self.reform(dA, max_len=self.max_len, embedding_dim=self.embedding_dim))
        dE = np.clip(dE, clip[0], clip[1])
        self.embedding_matrix -= self.lr * dE
        return dA


    def init_positional(self):
        """
        Initialise the PositionalEncoding object.
        """
        self.pos_encoder = PositionalEncoding(seqlen=self.max_len, output_dim=self.embedding_dim)


    @staticmethod
    def flatten(arr):
        """
        Flatten sentence embedding matrix to 2d.
        """
        return arr.reshape(arr.shape[0], -1)
    

    @staticmethod
    def reform(arr, max_len, embedding_dim):
        """
        Reshape a flattened array to its original dimensions.
        """
        return arr.reshape((arr.shape[0], max_len, embedding_dim))



class SeqNeuralNet:
    """
    Neural Net with an Embedding Layer, Positional Encodings.
    """

    loss_functions = {

        "log":[logloss, logloss_prime],
        "mse":[mse, mse_prime],
        "huber":[huber, huber_prime]
    }


    def __init__(self, layer_sizes: List[int], activations: List[str], lr: int = 0.1, loss:str = "mse", clip:List[float]=[-1.0, 1.0], seed: int or None = None, norm=False, vocab_size: int=None, embedding_dim:int=None, add_positional:bool=False, max_len=None):
        """
        Initialise the Sequential Neural Network.
        """
        if loss not in self.loss_functions.keys():
            raise ValueError(f"Loss function must be one of {list(self.loss_functions.keys())}")

        self.loss, self.loss_prime = self.loss_functions.get(loss)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.lr = lr
        self.seed = seed
        self.norm = norm
        self.layers = []
        self.losses = []
        self.clip = clip
        self.epochs = None
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.add_positional = add_positional
        self.max_len = max_len

        self.init_layers()

        self.evaluation = {}

    def __str__(self) -> str:
        outstr = ""
        params = sum([self.layer_sizes[i]*self.layer_sizes[i+1] for i in range(len(self.layer_sizes)-1)])
        outstr += f"## Architecture ##\nParams: {params}\nLayers: {len(self.layer_sizes)}"
        return outstr

    def init_layers(self):
        """
        Initialise the Layers from the specified parameters.
        """
        assert len(self.layer_sizes) == len(self.activations) + 1, f"{len(self.layer_sizes)},{len((self.activations))}\nMake sure len(layer_sizes) is 1 larger than len(activations).\nThe first layer size is n_features."

        if self.embedding_dim is not None and self.vocab_size is not None:
            self.layers += [EmbeddingLayer(embedding_dim=self.embedding_dim, 
                                           vocab_size=self.vocab_size,
                                           lr=self.lr,
                                           positional_encodings=self.add_positional,  
                                           max_len=self.max_len          
                                           )
                                ]
            
            self.layer_sizes[0] = self.embedding_dim * self.max_len

        for i in range(len(self.layer_sizes)-1):

            self.layers += [FFLayer(self.layer_sizes[i], 
                                    self.layer_sizes[i+1], 
                                    self.activations[i],
                                    lr = self.lr,
                                    seed = self.seed,
                                    )
                                ]
            
        [print(f"Layer {i}\n{self.layers[i]}") for i in range(len(self.layers))]

        
    def train(self, X: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, norm=False):
        """
        Train the network.
        """
        self.norm = norm
        losses = []
        durations = []

        for epoch in range(1, epochs + 1):
            
            epoch_start = datetime.now()

            epoch_loss = 0

            # Shuffle to avoid cycles in gradient descent
            permutation = np.random.permutation(X.shape[0])
            X = X[permutation]
            y = y[permutation]

            for batch_start in range(0, len(X), batch_size):
                batch_end = batch_start + batch_size
                batch_X = X[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]


                # Forward pass
                a = batch_X

                for layer in self.layers:
                    if norm and isinstance(layer, FFLayer):
                        a = layer.forward(a, norm=True)
                    else:
                        a = layer.forward(a)

                y_pred = a
    
                # Backward pass
                loss = self.loss(batch_y, y_pred)
                epoch_loss += loss

                dA = self.loss_prime(batch_y, y_pred)

                for layer in reversed(self.layers):
                    dA = layer.backprop(dA, clip=self.clip)
            
            epoch_end = datetime.now()
            durations.append(epoch_end-epoch_start)

            if (epoch == 1) or (epoch) % (epochs / 5) == 0:
                print(f"Epoch {epoch}/{epochs}: loss={epoch_loss / len(X)}")
                print(f"Mean Epoch Duration: {np.mean(durations)}")

            losses.append(epoch_loss)
            
        print(f"Total train time: {sum(durations)}")
        return losses


    def predict(self, X):
        """
        Inference given X where X is always a matrix.
        """

        A = X

        for layer in self.layers:
            if self.norm and isinstance(layer, FFLayer):
                A = layer.forward(A, norm=True)
            else:
                A = layer.forward(A)
        return A
    

    def multiclass_classification(self, X):
        """
        Wrapper for self.predict() for easy multiclass classification.
        """
        assert self.activations[-1] == "softmax"
        self.logits = self.predict(X)
        return np.argmax(self.logits,axis=1, keepdims=True)
    

    def binary_classification(self, X, threshold=0.5):
        """
        Wrapper for self.predict() for easy binary classification.
        """
        assert self.activations[-1] == "sigmoid"
        return np.where(self.predict(X) > threshold, 1, 0)
    
    def bounded_regression(self, X, bounds=[0,1]):
        """
        Wrapper for self.predict() for clipping output values to possible real set.
        """
        assert self.activations[-1] == "linear"
        return np.clip(self.predict(X), bounds[0], bounds[1])

    def save_pretrained(self, path="projects/text_adventure/models/net.pkl"):
        """
        Pickle the trained network.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
    

    @classmethod
    def from_pretrained(cls, path="projects/text_adventure/models/net.pkl"):
        """
        Load pre-trained pickled net.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

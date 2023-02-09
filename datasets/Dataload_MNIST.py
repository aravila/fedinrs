!pip install torch numpy

import collections
import numpy as np

np.random.seed(0)

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent

from tensorflow_federated import python as tff
from random import choices

n_epochs = 5

batch_size = 50

shuffle_buffer = 500

n_clients = 30

tf.compat.v1.enable_v2_behavior()

emnist_train, emnist_test = tff.simulation.dataset.emnist.loaddata()

def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([("x", tf.reshape(element["pixels"], [-1])),("y", tf.reshape(element["label"], [1]))])
    return dataset.repeat(n_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)

def gen_fed_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

sample_clients = emnist_train.client_ids[0: n_clients]
fed_train_data = gen_fed_data(emnist_train, sample_clients)

sample_clients_test = emnist_test.client_ids[0: n_clients]
fed_test_data = gen_fed_data(emnist_test, sample_clients_test)


            


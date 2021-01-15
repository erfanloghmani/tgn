import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from itertools import compress

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
# parser = argparse.ArgumentParser('TGN self-supervised training')
# parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
#                     default='wikipedia')
# parser.add_argument('--bs', type=int, default=200, help='Batch_size')
# parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
# parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
# parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
# parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
# parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
# parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
# parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
# parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
# parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
# parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
# parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
# parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
#                                                                   'backprop')
# parser.add_argument('--use_memory', action='store_true',
#                     help='Whether to augment the model with a node memory')
# parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
#   "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
# parser.add_argument('--message_function', type=str, default="identity", choices=[
#   "mlp", "identity"], help='Type of message function')
# parser.add_argument('--memory_updater', type=str, default="gru", choices=[
#   "gru", "rnn"], help='Type of memory updater')
# parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
#                                                                         'aggregator')
# parser.add_argument('--memory_update_at_end', action='store_true',
#                     help='Whether to update memory at the end or at the start of the batch')
# parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
# parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
#                                                                 'each user')
# parser.add_argument('--different_new_nodes', action='store_true',
#                     help='Whether to use disjoint set of new nodes for train and val')
# parser.add_argument('--uniform', action='store_true',
#                     help='take uniform sampling from temporal neighbors')
# parser.add_argument('--randomize_features', action='store_true',
#                     help='Whether to randomize node features')
# parser.add_argument('--use_destination_embedding_in_message', action='store_true',
#                     help='Whether to use the embedding of the destination node as part of the message')
# parser.add_argument('--use_source_embedding_in_message', action='store_true',
#                     help='Whether to use the embedding of the source node as part of the message')
# parser.add_argument('--dyrep', action='store_true',
#                     help='Whether to run the dyrep model')


# try:
#   args = parser.parse_args()
# except:
#   parser.print_help()
#   sys.exit(0)


BATCH_SIZE = 100
NUM_NEIGHBORS = 10
NUM_NEG = 1
NUM_EPOCH = 10
NUM_HEADS = 2
DROP_OUT = True
GPU = 0
DATA = 'nxpure_random'
NUM_LAYER = 1
LEARNING_RATE = 0.001
NODE_DIM = 100
TIME_DIM = 100
USE_MEMORY = True
MESSAGE_DIM = 100
MEMORY_DIM = 172
prefix = 'tgn-nxpr'

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = './saved_models/%s-%s.pth' % (prefix, DATA)
get_checkpoint_path = lambda x: './saved_checkpoints/%s-%s-%s.pth' % (prefix, DATA, x)

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
# logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=False, randomize_features=False)

num_users = len(set(full_data.sources))
num_items = len(set(full_data.destinations)) + 1

all_items = list(set(full_data.destinations))

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, False)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, False)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

user_embedding_static = torch.autograd.Variable(torch.eye(num_users).to(device))  # one-hot vectors for static embeddings
item_embedding_static = torch.autograd.Variable(torch.eye(num_items).to(device))  # one-hot vectors for static embeddings


# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# Initialize Model
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
          edge_features=edge_features, device=device,
          n_layers=NUM_LAYER,
          n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
          message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
          memory_update_at_start=not False,
          embedding_module_type='identity',
          message_function='identity',
          aggregator_type='last',
          memory_updater_type='gru',
          n_neighbors=NUM_NEIGHBORS,
          mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
          mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
          use_destination_embedding_in_message=False,
          use_source_embedding_in_message=False,
          dyrep=False)
tgn = tgn.to(device)

best_model_path = get_checkpoint_path(3)
tgn.load_state_dict(torch.load(best_model_path))


def sample_sequence_list(previous_interaction_idxs, k):
    idxs = np.random.randint(0, 2, size=(k, len(previous_interaction_idxs)))
    sample_sequences = []
    for i in range(k):
        sample_sequences.append(list(compress(previous_interaction_idxs, idxs[i, :])))
    return sample_sequences, idxs


def explain(model, data, interaction_idx, negative_edge_sampler, k=10):
    c_userid = data.sources[interaction_idx]
    previous_interaction_idxs = []
    for i in range(interaction_idx):
        if data.sources[i] == c_userid:
            previous_interaction_idxs.append(i)
    sampled_sequences, idxs = sample_sequence_list(previous_interaction_idxs, k)
    train_memory_backup = model.memory.backup_memory()
    with torch.no_grad():
        model = model.eval()

        all_predictions = []
        for c, seq in enumerate(sampled_sequences):
            print(c, seq)
            model.memory.restore_memory(train_memory_backup)
            cur_edge_idx = train_data.edge_idxs[-1] + 1
            seq.append(interaction_idx)
            for i, pidx in enumerate(seq):
                sources_batch = 0
                destinations_batch = data.destinations[pidx:pidx + 1]
                timestamps_batch = data.timestamps[pidx:pidx + 1]
                edge_idxs_batch = cur_edge_idx

                _, negative_samples = negative_edge_sampler.sample(1)

                pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                      negative_samples, timestamps_batch,
                                                                      edge_idxs_batch, NUM_NEIGHBORS)
                cur_edge_idx += 1
            all_predictions.append(pos_prob.cpu().detach().numpy())

    return all_predictions, idxs, previous_interaction_idxs

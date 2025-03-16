from .deal_dir import is_exist_dir
import torch
import numpy as np
import os
import random
from collections import OrderedDict
import torch.distributed as dist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def check_output_dir(path):
    if is_exist_dir(path):
        num = path[-1]
        if num.isdigit():
            res = int(num) + 1
            return path[:-1] + str(res)
        else:
            return path[:-1] + "_0"
    else:
        return path

def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True

# setup_seed(0)

def rebuild_model_state_dict(model_state_dict):
    new_state_dict = OrderedDict()
    for key, value in model_state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def init_distributed_mode(args):
    '''
    :param args: 
    :return:
    '''
#    os.environ can obtain some system-related information.
# Explanation of RANK: In a single-machine multi-GPU setup, running multiple GPUs can be understood as launching multiple processes, each handling one GPU. RANK represents the first process.
# For example, if you have two GPUs, RANK values will be 0 and 1. The first process has RANK = 0, and the second process has RANK = 1.
# Explanation of WORLD_SIZE: The number of GPUs used or the number of processes.
# Explanation of LOCAL_RANK: The GPU index within a process, not an explicit parameter, but assigned internally by torch.distributed.launch.
# For example, rank = 3, local_rank = 0 means the first GPU in the third process.
# This may seem confusing, but you’ll understand once you use it.
# These parameters help with training and managing output. For example, when multiple processes print information, it can be redundant.
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.device = int(os.environ['LOCAL_RANK'])
    
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.device = args.rank % torch.cuda.device_count()
    else:
        # print("NOT using distributed mode")
        raise EnvironmentError("NOT using distributed mode")
        # return
    # print(args)
    # print("1111111-")
    #
    args.distributed = True

    torch.cuda.set_device(args.device)
    print('gpu',args.device, args.rank)
    # print("33333-")
    # 这里是GPU之间的通信方式，有好几种的，nccl比较快也比较推荐使用。
    args.dis_backend = 'nccl'
    # 启动多GPU
    dist.init_process_group(
        backend=args.dis_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    # print("2222222-")

    dist.barrier()

def print_heatmap(fc, cmap='RdYlBu_r', square=True, xticklabels=10, yticklabels=10, xrotation=0, yrotation=0,
                  annot=False, show=True, save=None,
                  figsize=None, cbar=True, bbox_inches=None):
    # sns.heatmap(fc, cmap='YlGnBu_r', square=True, xticklabels=10,yticklabels=10)
    plt.figure(figsize=figsize)
    sns.heatmap(fc, cmap=cmap, square=square, xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, cbar=cbar)
    # mask = np.zeros_like(fc)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(fc, cmap='YlGnBu_r', square=True, xticklabels=10, vmax=1, vmin=-1)
    plt.xticks(rotation=xrotation)
    plt.yticks(rotation=yrotation)
    if save:
        plt.savefig(save, bbox_inches=bbox_inches)
    if show:
        plt.show()
    else:
        plt.close()

def print_cluster(X, labels):
    labels_unique = np.unique(labels)
    n_cluters = len(labels_unique)
    fc_embedded = TSNE().fit_transform(X)
    for i in range(n_cluters):
        members = labels == i
        plt.scatter(X[members, 0], X[members, 1], label=i, marker='o')
    # plt.scatter(X[:, 0], X[:, 1], label=['red', 'green', 'blue'], c=labels, marker='o')
    plt.legend()
    plt.show()

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_epochs(fname, X, epochs, xlabel, ylabel, legends, max=True):
    plt.figure()
    for i, x in enumerate(X):
        val = np.max(x) if max else np.min(x)
        idx = np.argmax(x) + 1 if max else np.argmin(x) + 1
        plt.plot(epochs, x, label=legends[i])
        plt.plot(idx, val, 'ko')
        plt.annotate(f'({idx},{val:.4f})', xy=(idx, val), xytext=(idx, val))

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    # plt.show()
    plt.close()
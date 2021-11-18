from experiment.simclr.main import run_train

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')

    parser.add_argument('--feature_dim',
                        default=128,
                        type=int,
                        help='Feature dim for latent vector')
    parser.add_argument('--temperature',
                        default=0.5,
                        type=float,
                        help='Temperature used in softmax')
    parser.add_argument(
        '--k',
        default=200,
        type=int,
        help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size',
                        default=512,
                        type=int,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs',
                        default=500,
                        type=int,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--loss', default='ce', type=str, help='loss function')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url',
                        default='tcp://localhost:10001',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument(
        '--multiprocessing-distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training')

    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        args.world_size = args.world_size * ngpus_per_node
        mp.spawn(run_train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        run_train(args.gpu, ngpus_per_node, args)

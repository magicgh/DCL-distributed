import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment.simclr import utils
from experiment.simclr.model import Model

# train for one epoch to learn unique features
from loss import DCL
from loss.dcl import DCLW


def train(net, data_loader, train_optimizer, args, epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(
            non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        if args.loss == 'dcl':
            l = DCL(temperature=args.temperature)
            loss = l(out_1, out_2)
        elif args.loss == 'dclw':
            l = DCLW(temperature=args.temperature)
            loss = l(out_1, out_2)
        elif args.loss == 'ce':
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(
                torch.mm(out,
                         out.t().contiguous()) / args.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(
                2 * args.batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(
                2 * args.batch_size, -1)

            # compute loss
            pos_sim = torch.exp(
                torch.sum(out_1 * out_2, dim=-1) / args.temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        train_bar.set_description(
            'Rank: {} Train Epoch: [{}/{}] Loss: {:.4f}'.format(
                args.rank, epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, args, epoch):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(
                memory_data_loader,
                desc=f'Rank {args.rank} Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets,
                                      device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(
                non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1),
                                      dim=-1,
                                      index=sim_indices)
            sim_weight = (sim_weight / args.temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.k,
                                        args.c,
                                        device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1,
                                                  index=sim_labels.view(-1, 1),
                                                  value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(
                one_hot_label.view(data.size(0), -1, args.c) *
                sim_weight.unsqueeze(dim=-1),
                dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(
                dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(
                dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description(
                'Rank {} Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(
                    args.rank, epoch, args.epochs,
                    total_top1 / total_num * 100,
                    total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def run_train(gpu, ngpus_per_node, args):

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + args.gpu
        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=args.rank)
    torch.distributed.barrier()

    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        args.master = True
    else:
        args.master = False

    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    
    train_data = utils.CIFAR10Pair(root='data',
                                    train=True,
                                    transform=utils.train_transform,
                                    download=True)
    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=(train_sampler is None),
                                num_workers=16,
                                pin_memory=True,
                                sampler=train_sampler,
                                drop_last=True)

    memory_data = utils.CIFAR10Pair(root='data',
                                    train=True,
                                    transform=utils.test_transform,
                                    download=True)
    memory_loader = DataLoader(memory_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True)
    test_data = utils.CIFAR10Pair(root='data',
                                    train=False,
                                    transform=utils.test_transform,
                                    download=True)
    test_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True)

    # model setup and optimizer config

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = Model(feature_dim).cuda(args.gpu)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
    else:
        model = Model(feature_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    args.c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}'.format(feature_dim, temperature, k,
                                               batch_size, epochs, args.loss)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train(model, train_loader, optimizer, args, epoch)
        
        
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, args,
                                      epoch)

        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        # save statistics
        if args.master:
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(
                'results/{}_statistics.csv'.format(save_name_pre),
                index_label='epoch')
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                torch.save(model.state_dict(),
                           'results/{}_model.pth'.format(save_name_pre))

import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')
from distributed_utils import dist_init, average_gradients, broadcast_params
from torch.utils.data.distributed import DistributedSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from read_dual import *
from model_deeplabv2 import model_triplet
from utils_triplet_val import Trainer
import logging
from argparse import ArgumentParser
import torch.distributed as dist
import json


# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Multi-label classification')
    parser.add_argument('--config', type=str, default='', help='Hyper parameters config file')
    parser.add_argument('--data_root', type=str, default='', help='data root')
    parser.add_argument('--train_list', type=str, help='train list')
    parser.add_argument('--val_list', type=str, help='val list')

    parser.add_argument('--syncbn', type=int, default=1, help='adopt syncbn or not')
    parser.add_argument('--gpu', type=int, default=[0, 1, 2, 3], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=1, help='data loader workers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=1,
                        help='batch size for validation during training, memory and speed tradeoff')

    parser.add_argument('--base_lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--power', type=float, default=0.9, help='power in poly learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--save_step', type=int, default=10, help='model save step (default: 10)')
    parser.add_argument('--save_path', type=str, default='tmp', help='model and summary save path')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrained model (default: none)')
    parser.add_argument('--evaluate', type=int, default=0, help='evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend')
    parser.add_argument('--weighted', type=int, default=0, help='the classes weighted ')
    parser.add_argument('--root_be', type=str, default='', help='the classes weighted ')
    parser.add_argument('--root_af', type=str, default='', help='the classes weighted ')
    parser.add_argument('--root_mask', type=str, default='', help='the classes weighted ')


    """
    for dist part
    """

    parser.add_argument('--bn_group', type=int, default=4, help='the total GPUs used ')
    parser.add_argument('--dist', dest='dist', type=int, default=1,
                        help='distributed training or not')
    parser.add_argument('--backend', dest='backend', type=str, default='nccl',
                        help='backend for distributed training')
    parser.add_argument('--port', dest='port', required=True,
                        help='port of server')

    return parser


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



BATCH_SIZE=16


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path):
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location = lambda storage, loc: storage.cuda(device))
    if 'weight' in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['weight'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    model.load_state_dict(pretrained_dict, strict=False)
    return model




def main(batch_size):
    args = get_parser().parse_args()
    logger = get_logger()

    if args.dist:
        dist_init(args.port, backend=args.backend)
    if len(args.gpu) == 1:
        args.syncbn = False

    world_size = 1
    rank = 0
    if args.dist:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if rank == 0:
        logger.info('dist:{}'.format(args.dist))
        logger.info("=> creating model ...")

    if args.bn_group > 1:
        args.syncbn = True
    else:
        args.syncbn = False

    model = model_triplet()

    '''if rank == 0:
        logger.info(model)'''
    model=model.cuda()
    if args.pretrain:
        if(rank==0):
            logger.info("model loads the pretrain model[{}]".format(args.pretrain))
        model=load_pretrain(model, args.pretrain)
    cudnn.enabled = True
    cudnn.benchmark = True
    optimizer = optim.Adam(params=model.parameters())

    if args.dist:
        broadcast_params(model)
        #model = model.cuda()
        #model = torch.nn.parallel.DistributedDataParallel(model)

    train_data=ImageFolder(args.train_list, args.root_be, args.root_af, args.root_mask)
    val_data=ImageFolder_val(args.val_list,args.root_be,args.root_af,args.root_mask)

    val_sampler = None
    train_sampler = None
    if args.dist:
        val_sampler = DistributedSampler(val_data)
        train_sampler = DistributedSampler(train_data)
    train_loader=torch.utils.data.DataLoader(train_data,
        shuffle=False if train_sampler else True,num_workers=args.workers,
        batch_size=args.batch_size,pin_memory=False,sampler=train_sampler)
    validation_loader=torch.utils.data.DataLoader(val_data,
        batch_size=args.batch_size_val, shuffle=False,
        pin_memory=False,sampler=val_sampler,num_workers=args.workers)


    trainer = Trainer(model, optimizer, nn.BCELoss,logger,args,rank)
    trainer.loop(args.epochs, train_loader, validation_loader)


if __name__ == '__main__':
    main(BATCH_SIZE)
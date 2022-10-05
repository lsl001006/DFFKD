import argparse
import os
import random
import time
import warnings

from datetime import datetime
from mosaic_core import registry
from mosaic_core import engine
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import time
import logging
from torch.utils.tensorboard import SummaryWriter

from PIL import PngImagePlugin
from mosaic_core.pipeline.FL import MultiTeacher, validate

from mosaic_core.pipeline.FL import OneTeacher

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


parser = argparse.ArgumentParser(description='MosaicKD for OOD data')

# path
parser.add_argument('--DATAPATH', default='data/fed', type=str)
parser.add_argument('--ckpt_path', default="/data/repo/code/1sl/DFFK/checkpoints/", type=str,
                    help="location to store training checkpoints")
parser.add_argument('--gen_ckpt_path',default="./ckpt",type=str,
                    help='location to save generator and discriminator and pure student checkpoint')

# basic training settings
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='resnet8')
parser.add_argument('--pipeline', default='multi_teacher',help='mosaickd, multi_teacher')
parser.add_argument('--n_classes', '--num_classes', type=int, default=10)
parser.add_argument('--N_PARTIES', '--n_parties', type=int, default=20)
parser.add_argument('--local_percent', type=float, default=1.0)
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--unlabeled', default='cifar10')
parser.add_argument('--log_tag', default='')
parser.add_argument('--logfile', default='', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--fp16', action='store_true',help='use fp16 to accelerate training')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--from_teacher_ckpt', default="", type=str, help='path used to load pretrained teacher ckpts')
parser.add_argument('--use_pretrained_generator', default='', type=str, help='use pretrained generator instead of conventional Generator')
parser.add_argument('--use_maxIters', action='store_true',
                    help='use max to calculate iters_per_round instead of min therefore taking full usage of local data')

# pate differential privacy
parser.add_argument('--use_pate', action='store_true',help='use pate to preserve differential privacy')
parser.add_argument('--num_moments', default=100, help="Number of higher moments to use for epsilon calculation for pate-gan")
parser.add_argument('--target-delta', type=float, default=1e-5, help='Delta differential privacy parameter')
parser.add_argument('--lap_scale',type=float,default=0.0001, 
                    help='Inverse laplace noise scale multiplier. '
                    ' A larger lap_scale will reduce the noise'
                    ' that is added per iteration of training.')
parser.add_argument('--onehot_vote', action='store_true',help='use one-hot vote instead of logits')
parser.add_argument('--add_noise', action='store_true',help='add laplacian noise to vote results')
parser.add_argument('--local_weight', action='store_true',help='use local weight to vote in addition to local cls weight')
parser.add_argument('--local_cls_weight', action='store_true',help='use local cls weight when voting')
parser.add_argument('--max_vote', action = 'store_true',help='use max vote instead of half agreement')                    

# gradient clip + noise
parser.add_argument('--clip', action='store_true',help='use clip to discriminator gradients')
parser.add_argument('--noise_multiplier', '-noise', type=float, default=1.07, help='noise multiplier')

# Learning Rate
parser.add_argument('--lr', '--learning_rate', default=5e-3, type=float,metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_min', default=1e-4, type=float,help='min learning rate')
parser.add_argument('--lr_g', default=1e-3, type=float)
parser.add_argument('--optimizer', type=str, default='SGD',help='SGD or ADAM')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('--modify_optim_lr', action='store_true',help='modify optimizer lr when resuming model-trainning process')
parser.add_argument('--fixed_lr', action='store_true',help='Use fixed Learning Rate while training')

# hyper params
parser.add_argument('--T', default=1.0, type=float,help="Distillation temperature. T > 10000.0 will use MSELoss")
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--ngf', default=64, type=int,help='modify the feature map size of generator')
parser.add_argument('--z_dim', default=100, type=int)
parser.add_argument('--output_stride', default=1, type=int)
# parser.add_argument('--align', default=1, type=float)
# parser.add_argument('--local', default=1, type=float)
# parser.add_argument('--adv', default=1.0, type=float)
# parser.add_argument('--balance', default=10.0, type=float)

# weights
parser.add_argument('--w_disc', type=float, default=1.0)
parser.add_argument('--w_gan', type=float, default=1.0)
parser.add_argument('--w_adv', type=float, default=1.0)
parser.add_argument('--w_algn', type=float, default=1.0)
parser.add_argument('--w_baln', type=float, default=10.0)
parser.add_argument('--w_dist', type=float, default=1.0)



# parser.add_argument('-p', '--print_freq', default=0, type=int,metavar='N', help='print frequency (default: 10)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
# parser.add_argument('--ood_subset', action='store_true',
#                     help='use ood subset')
# parser.add_argument('--world_size', default=-1, type=int,
#                     help='number of nodes for distributed training')

# parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist_backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--use_l1_loss', action='store_true',
#                     help='default use kldiv, using this command will use l1_loss')

# rarely used
parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
parser.add_argument('--is_emsember_generator_GAN_loss', default='y', type=str)
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--gen_loss_avg', action='store_true',help='use average instead of ensemble locals')
parser.add_argument('--save_img', action='store_true',help='save_img every time update student')



best_acc1 = 0
def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    #############################logs settings##############################
    handlers = [logging.StreamHandler()]
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    if args.logfile:
        args.logfile = f'{datetime.now().strftime("%m%d%H%M")}'+args.logfile
        writer = SummaryWriter(comment=args.logfile)
        handlers.append(logging.FileHandler(
            f'./logs/{args.logfile}.txt', mode='a'))
    else:
        args.logfile = 'debug'
        writer = None
        handlers.append(logging.FileHandler(f'./logs/debug.txt', mode='a'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,
    )
    logging.info(args)
    ############################## > main function < ##############################
    main_worker(args.gpu, ngpus_per_node, args, writer)


def main_worker(rank, ngpus_per_node, args, writer):
    args.gpu = rank
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    

    ###################################logs###################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = '%s-%s-%s' % (args.dataset, args.teacher,
                             args.student) if args.multiprocessing_distributed else '%s-%s-%s' % (args.dataset, args.teacher, args.student)
    args.logger = engine.utils.logger.get_logger(log_name,
                                                 output='checkpoints/MosaicKD/log-%s-%s-%s-%s%s.txt' % (args.dataset, args.unlabeled, args.teacher, args.student, args.log_tag))
    
    args.tb = SummaryWriter(log_dir=os.path.join(
        'tb_log', log_name+'_%s' % (time.asctime().replace(' ', '-'))))


    if args.rank <= 0:
        for k, v in engine.utils.flatten_dict(vars(args)).items():  # print args
            args.logger.info("%s: %s" % (k, v))
    
    ###############################setup models################################
    teacher, student, netG, netD, normalizer = setup_models(args)
    if args.pipeline == "mosaickd":
        # pass
        pipeline = OneTeacher(student, teacher, netG, netD, args, writer)
        pipeline.update()
    ###############################Entrance#####################################
    elif args.pipeline == "multi_teacher":
        pipeline = MultiTeacher(student, teacher, netG, netD, args, writer)
        pipeline.update()

    ################################# END ##################################


def setup_models(args):
    ############################################
    # Setup Models
    ############################################
    student = registry.get_model(args.student, num_classes=args.n_classes)

    teacher = registry.get_model(
        args.teacher, num_classes=args.n_classes, pretrained=True).eval()
    if args.from_teacher_ckpt == '':
        teacher.load_state_dict(torch.load('mosaic_core/checkpoints/pretrained/%s_%s.pth' %
                                (args.dataset, args.teacher), map_location='cpu')['state_dict'])

    normalizer = engine.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    args.normalizer = normalizer
    netG = engine.models.generator.Generator(nz=args.z_dim, ngf=args.ngf, nc=3, img_size=32)
    netD = engine.models.generator.PatchDiscriminator2(nc=3, ndf=128)

    ############################################
    # Device preparation
    ############################################
    torch.cuda.set_device(args.gpu)
    student = student.cuda(args.gpu)
    teacher = teacher.cuda(args.gpu)
    netG = netG.cuda(args.gpu)
    netD = netD.cuda(args.gpu)
    return teacher, student, netG, netD, normalizer


# def prepare_ood_data(train_dataset, model, ood_size, args):
#     model.eval()
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=False,
#         num_workers=args.workers)
#     if os.path.exists('checkpoints/ood_index/%s-%s-%s-ood-index.pth' % (args.dataset, args.unlabeled, args.teacher)):
#         ood_index = torch.load('checkpoints/ood_index/%s-%s-%s-ood-index.pth' %
#                                (args.dataset, args.unlabeled, args.teacher))
#     else:
#         with torch.no_grad():
#             entropy_list = []
#             model.cuda(args.gpu)
#             model.eval()
#             for i, (images, target) in enumerate(tqdm(train_loader)):
#                 if args.gpu is not None:
#                     images = images.cuda(args.gpu, non_blocking=True)
#                 if torch.cuda.is_available():
#                     target = target.cuda(args.gpu, non_blocking=True)
#                 # compute output
#                 output = model(images)
#                 p = torch.nn.functional.softmax(output, dim=1)
#                 ent = -(p*torch.log(p)).sum(dim=1)
#                 entropy_list.append(ent)
#             entropy_list = torch.cat(entropy_list, dim=0)
#             ood_index = torch.argsort(entropy_list, descending=True)[
#                 :ood_size].cpu().tolist()
#             model.cpu()
#             os.makedirs('checkpoints/ood_index', exist_ok=True)
#             torch.save(ood_index, 'checkpoints/ood_index/%s-%s-%s-ood-index.pth' %
#                        (args.dataset, args.unlabeled, args.teacher))
#     return ood_index



if __name__ == '__main__':
    main()

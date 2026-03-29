import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

import argparse

#os.system('wandb login xxx')
#import wandb
from time import time, strftime
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
from kdutils import seed_all, GradualWarmupScheduler
#from torchvision.models.resnet import resnet18
from torchvision import transforms
from models import *
from models.spike_layer import LIFAct
from functions import tmpr_loss
import models.spike_layer
from tqdm import tqdm
from data.autoaugment import CIFAR10Policy, Cutout
from data.cifar_dvs import DVSCifar10
import logging
#from models import ImageNet_cnn, ImageNet_snn
#from loss_kd import feature_loss,  logits_loss
#from spikingjelly.clock_driven import functional
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# def get_model(name):
#     return func_dict[name]


parser = argparse.ArgumentParser(description='Ternary_SNN_Training')
# parser.add_argument("--datapath", type=str, default='/home/xlab/xdata/imagenet/')
parser.add_argument('--dataset', type = str, choices = ['cifar10', 'cifar100', 'cifar10-dvs', 'imagenet100'], default = 'cifar10')
parser.add_argument('--complementary', action='store_true', default=False)
parser.add_argument('--model', type=str, default='resnet20m')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--warm_up', action='store_true', default=False)
parser.add_argument('--load_weight', action='store_true', default=False)
parser.add_argument("--feature_epochs", type=int, default=10)
parser.add_argument('--record_mem_dis', type=int, default=0)
parser.add_argument('--spike', type=int, default=1, help='use spiking network')
parser.add_argument('--V_th', type=float, default=0.5, help='spike threshold')
parser.add_argument('--tau', type=float, default=0.25, help='membrane time constant')
parser.add_argument('--loss', type=str, default='ce', help='loss function')
parser.add_argument('--lamb', default=1.0, type=float, help='membrane loss weight')
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'], help='optimizer')
parser.add_argument('--lr', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
parser.add_argument('--decay_parameters', action='store_true', default=False, help='Whither to decay the parameters.')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--step', default=4, type=int, help='snn step')
parser.add_argument('--parallel', action='store_true', default=False,help='Whither to use multiple GPUs.')
parser.add_argument('--gpu',type=str,default='0',help='GPU(s) ID. When using parallel training, the IDs must be specified as a string of comma-separated integers, like 0-1-2-3. Default: 0.')
parser.add_argument('--seed', type=int, default=1000)
args = parser.parse_args()

######## init seed ########
seed_all(args.seed)

if args.parallel:
    gpus=args.gpu.split('-')
    os.environ['CUDA_VISIBLE_DEVICES']=gpus[0] if len(gpus)==1 else ','.join(args.gpu.split('-'))
    device=torch.device('cuda')
else:
    device=torch.device(f'cuda:{args.gpu}')

# torch.distributed.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)

######## load dataset and input model########
experiment_path=os.path.dirname(os.path.abspath(__file__))+f'/{args.dataset}'

if args.complementary:
    complementary = 'static'
else:
    complementary = None

if args.dataset == 'cifar10':
    normalization_mean = (0.4914, 0.4822, 0.4465)
    normalization_std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(normalization_mean, normalization_std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])
    dataset = torchvision.datasets.CIFAR10
    train_data = dataset(experiment_path + '/data', train=True, transform=transform_train, download=True)
    test_data = dataset(experiment_path + '/data', train=False, transform=transform_test, download=False)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    num_classes = 10
    input_channels = 3
    # args.decay_parameters = True

elif args.dataset == 'cifar100':
    normalization_mean = (0.5071, 0.4867, 0.4408)
    normalization_std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(normalization_mean, normalization_std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])
    dataset = torchvision.datasets.CIFAR100
    train_data = dataset(experiment_path + '/data', train=True, transform=transform_train, download=True)
    test_data = dataset(experiment_path + '/data', train=False, transform=transform_test, download=False)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    num_classes = 100
    input_channels = 3
    # args.decay_parameters = True

elif args.dataset == 'imagenet100':
    normalization_mean = (0.485, 0.456, 0.406)
    normalization_std = (0.229, 0.224, 0.225)
    train_dataset = torchvision.datasets.ImageFolder(
        experiment_path + '/data/train',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std)])
    )
    test_dataset = torchvision.datasets.ImageFolder(
        experiment_path + '/data/val',
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std)])
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    num_classes = 100
    input_channels = 3
    # args.decay_parameters = True

elif args.dataset == 'cifar10-dvs':
    if args.complementary:
        complementary = 'dvs'
    dataset = DVSCifar10
    train_data = dataset(experiment_path + '/data', train=True, transform=True)
    test_data = dataset(experiment_path + '/data', train=False, transform=False)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    num_classes = 10
    input_channels = 2
    args.step = 10
    # args.decay_parameters = True

connect_f='ADD'
if args.model == 'resnet19':
    model = resnet19_cifar(num_classes=num_classes, input_c=input_channels)
elif args.model == 'resnet20':
    model = resnet20_cifar(num_classes=num_classes, input_c=input_channels)
elif args.model == 'resnet20m':
    model = resnet20_cifar_modified(num_classes=num_classes, input_c=input_channels)
elif args.model == 'resnet32':
    model = resnet32_cifar(num_classes=num_classes, input_c=input_channels)
elif args.model == 'resnet34':
    model = ResNet34(num_classes=num_classes, input_c=input_channels)
elif args.model == 'sewresnet34':
    model = SEWResNet34(num_classes=num_classes, input_c=input_channels, connect_f=connect_f)
elif args.model == 'vggsnn':
    model = vggsnn(num_classes=num_classes, input_c=input_channels, dropout=args.dropout)
elif args.model == 'vgg11':
    model = vgg11_bn(num_classes=num_classes, input_c=input_channels, dropout=args.dropout)
elif args.model == 'vgg16':
    model = vgg16_bn(num_classes=num_classes, input_c=input_channels, dropout=args.dropout)

######## save model #######
time_str=strftime(r'%Y-%m-%d_%H-%M-%S')
if not os.path.exists(experiment_path+'/result'):
    os.mkdir(experiment_path+'/result')
os.mkdir(f'{experiment_path}/result/T{args.step}_{args.model}_{time_str}/')
model_save_name = f'{experiment_path}/result/T{args.step}_{args.model}_{time_str}/weight/snn-{args.model}-{args.step}-{args.epoch}.pth'
log_file_path = f'{experiment_path}/result/T{args.step}_{args.model}_{time_str}/logs'
os.mkdir(log_file_path)
os.mkdir(f'{experiment_path}/result/T{args.step}_{args.model}_{time_str}/weight')

######## init logger #######
logging.basicConfig(level=logging.INFO, filename=log_file_path+'/train.log', filemode='w')


######## load weight #######
#model.load_state_dict(torch.load('raw/ann-resnet18.pth', map_location='cpu'))

######## change to snn #######
if args.spike:
    model = SpikeModel(model, args.step, args.V_th, args.tau, complementary, True if args.loss == 'tmpr' else False)
    model.set_spike_state(True)
    
######## init bias #######    
#model = init_bias(model)    
    
SNN = model.to(device)

######## show parameters #######
n_parameters = sum(p.numel() for p in SNN.parameters() if p.requires_grad)
print('number of params:', n_parameters)
print(SNN)
logging.info(f'number of params:{n_parameters}\n')
logging.info(SNN)
for arg in args._get_kwargs():
    logging.info(f'{arg[0]}={arg[1]}')

if args.parallel:
    SNN = torch.nn.DataParallel(SNN)
# ######## parallel #######
# SNN = torch.nn.SyncBatchNorm.convert_sync_batchnorm(SNN)
# SNN = torch.nn.parallel.DistributedDataParallel(SNN, device_ids=[[args.local_rank]],output_device=[args.local_rank], find_unused_parameters=False)
# model_without_ddp = SNN.module

######## amp #######
require_list = False
if args.loss == 'tmpr':
    train_loss_fun = tmpr_loss
    test_loss_fun = torch.nn.CrossEntropyLoss().to(device)
    require_list = True
    handles = []
    # mem_list = []
    neuron_hook = lambda layer_name: lambda module, input, output: mem_list.append(torch.stack(module.mem, 0).flatten(1, -1))
    for n, m in SNN.named_modules():
        if isinstance(m, LIFAct):
            handles.append(m.register_forward_hook(neuron_hook(n)))
elif args.loss =='ce': 
    train_loss_fun = torch.nn.CrossEntropyLoss().to(device)
    test_loss_fun = torch.nn.CrossEntropyLoss().to(device)
scaler = GradScaler(device=device)

######## split BN #######

parameters = split_weights(SNN, args.decay_parameters)
if args.optimizer == 'sgd':
    optimer = torch.optim.SGD(params=parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimer = torch.optim.Adam(params=parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimer = torch.optim.AdamW(params=parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
# optimer = torch.optim.AdamW(params=SNN.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3)

scheduler = CosineAnnealingLR(optimer, T_max=args.epoch, eta_min=0)
scheduler_warm = None
if args.warm_up:
    scheduler_warm = GradualWarmupScheduler(optimer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

######## init tensorboard #######
writer=SummaryWriter(log_file_path)

best_acc = 0.0
sta = time()

if __name__ == '__main__':

    for i in range(args.epoch):
        train_loss_ce_all = 0
        test_loss_ce_all = 0
        train_loss_ce_mean = 0
        test_loss_ce_mean = 0
        start_time = time()
        right = 0
        SNN.train()
        for step, (imgs, target) in enumerate(tqdm(train_loader)):
            if require_list:
                mem_list = []
            imgs, target = imgs.float().to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                output = SNN(imgs, is_drop=False)
                if args.loss == 'tmpr':
                    loss = train_loss_fun(output, target, mem_list, args.step, args.lamb)
                elif args.loss == 'ce':
                    loss = train_loss_fun(output, target)

            right = (output.argmax(1) == target).sum() + right
            train_loss_ce_all += loss.item()

            optimer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimer)
            scaler.update()

            # if step % 100 == 0 and args.local_rank == 0:
            #     print("step:{:.2f} loss_ce:{:.2f}".format(step / len(train_data), loss.item()))
        accuracy1 = right / len(train_loader.dataset)
        train_loss_ce_mean = train_loss_ce_all / len(train_loader.dataset)
        if args.warm_up:
            scheduler_warm.step()
        else:
            scheduler.step()

        SNN.eval()
        right = 0

        with torch.no_grad():
            for (imgs, target) in tqdm(test_loader):
                if require_list:
                    mem_list = []
                imgs, target = imgs.float().to(device, non_blocking=True), target.to(device, non_blocking=True)
                output  = SNN(imgs, is_drop=False)
                right = (output.argmax(1) == target).sum() + right
                loss = test_loss_fun(output, target)
                test_loss_ce_all += loss.item()

            accuracy = right / len(test_loader.dataset)
            test_loss_ce_mean = test_loss_ce_all / len(test_loader.dataset)
            end_time = time()
        print("epoch:{} time:{:.0f} | train_loss:{:.4f} | train_acc:{:.4f} | test_loss:{:.4f} | test_acc:{:.4f} | eta:{:.2f}".format(i+1,
            end_time - start_time,train_loss_ce_all,accuracy1,test_loss_ce_all,accuracy, (end_time - start_time) * (args.epoch - i - 1) / 3600))
        logging.info("epoch:{} time:{:.0f} | train_loss:{:.4f} | train_acc:{:.4f} | test_loss:{:.4f} | test_acc:{:.4f} | eta:{:.2f}".format(i+1,
            end_time - start_time,train_loss_ce_all,accuracy1,test_loss_ce_all,accuracy, (end_time - start_time) * (args.epoch - i - 1) / 3600))
        if accuracy > best_acc:
            best_acc = accuracy
            print("best_acc:{:.4f}".format(best_acc))
            if isinstance(SNN,torch.nn.DataParallel):
                torch.save(SNN.module.cpu().state_dict(), model_save_name)
            else:
                torch.save(SNN.cpu().state_dict(),model_save_name)
            SNN.to(device)
            #print({"test_acc": accuracy, "train_acc": accuracy1, "loss_ce_all": loss_ce_all, 'epoch': i, })
        writer.add_scalar('test_acc', accuracy, i+1)
        writer.add_scalar('train_acc', accuracy1, i+1)
        writer.add_scalar('train_loss_all', train_loss_ce_all, i+1)
        writer.add_scalar('test_loss_all', test_loss_ce_all, i+1)
        writer.add_scalar('train_loss_mean', train_loss_ce_mean, i+1)
        writer.add_scalar('test_loss_mean', test_loss_ce_mean, i+1)

    end = time()
    print(end - sta)
    print("best_acc:{:.4f}".format(best_acc))
    logging.info(end - sta)
    logging.info("best_acc:{:.4f}".format(best_acc))

    writer.close()
    if args.loss == 'tmpr':
        for handle in handles:
            handle.remove()
    
#python3 -m torch.distributed.launch --nproc_per_node 8 --nnode 1 --master_port=25641 Train.py --spike --step 4

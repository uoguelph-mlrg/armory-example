"""
reproduce_resisc45_keras_baseline.py -- Reproduce RESISC45 Keras baseline in Pytorch.
"""
# general
import os
import os.path as osp
import subprocess
import csv
import time
import argparse
import numpy as np

# ml libs
import torch
import torch.nn as nn

from torchvision import models

from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary

# my utils
#from task_3_utils import (save_checkpoint, save_amp_checkpoint, evaluate, adjust_learning_rate)
#from apex import amp

# ARMORY
from armory import paths
from armory.data import datasets
from armory.data import adversarial_datasets

import sys
sys.path.append("../../")
from uog_models.pytorch.densenet121_resisc45 import mean_std, preprocessing_fn, resisc_densenet121

from train_utils import evaluate_loss_accuracy, save_checkpoint

if __name__ == '__main__':

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
        )

    parser = argparse.ArgumentParser()

    # dataset, admin, checkpointing and hw details
    parser.add_argument('--dataroot', help='path to dataset', type=str,
                        default=paths.HostPaths().dataset_dir)
    
    parser.add_argument('--logdir', help='directory to store checkpoints; \
                        if None, nothing will be saved')
    
    parser.add_argument("--resume", default="", type=str,
                        help="path to latest checkpoint (default: none)")
    
    parser.add_argument('--do_print', help="print ongoing training progress",
                        action="store_true")
    
    parser.add_argument('--pretrained', help="pre-train model on ImageNet?", action="store_true")
    
    parser.add_argument('--gpu', help='physical id of GPU to use')
    
    parser.add_argument('--seed', help='random seed', type=int, default=1)

    # model arch and training meta-parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: fcn_resnet50)')
    
    parser.add_argument('--opt', default='adam', choices=['adam', 'sgd'], help='optimizer')
    
    parser.add_argument('--epochs', help='number of epochs to train for', type=int, default=1)
    
    #parser.add_argument('--drop', help='epoch to first drop the initial \ learning rate', type=int, default=30)
    
    parser.add_argument('--bs', help='SGD mini-batch size', type=int, default=32)
    parser.add_argument('--lr', help='initial learning rate', type=float, default=1e-3)
    parser.add_argument('--wd', help='weight decay regularization', type=float, default=0)
    parser.add_argument('--fp16', help='use apex to train with fp16 parameters', action="store_true")
    #parser.add_argument('--tag', help='custom tag to ID debug runs')
    
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if torch.cuda.is_available():
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(args.seed)

    gitcommit = subprocess.check_output(['git', 'rev-parse', '--short',
                                         'HEAD']).decode('ascii').strip()

    arch = ''
    arch += 'pretrained_' if pretrained else ''
    save_path = osp.join(
        args.logdir, arch + 'opt%s/lr%.e/wd%.e/bs%d/ep%d/seed%d/%s' % (args.opt, args.lr, args.wd,
                                                                       args.bs,
                                                                       args.epochs,
                                                                       args.seed,
                                                                       gitcommit))
    print('Saving model to ', save_path)

    ckpt_name = args.arch + '_lr%.e_wd%.e_bs%d_ep%d_seed%d' % (args.lr, args.wd, args.bs, args.epochs, args.seed)

    # Logging stats
    result_folder = osp.join(save_path, 'results/')
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    #model_string = args.arch + '_bs%d' % args.bs + '_wd%.e' % args.wd + '_' + \
    #args.sess + '_' + str(args.seed)
    logname = osp.join(result_folder, ckpt_name + '.csv')

    if not osp.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'lr', 'train loss', 'val loss'])

    # Prepare datasets
    ds_train = datasets.resisc45(
        split_type='train', 
        epochs=args.epochs, 
        batch_size=args.bs, 
        dataset_dir=args.dataroot,
        preprocessing_fn=preprocessing_fn,
        framework='numpy')
    
    def evaluate_test_metrics(net, loss_fn):
        
        ds_test = datasets.resisc45(
            split_type='test', epochs=1, 
            batch_size=args.bs, 
            dataset_dir=args.dataroot,
            preprocessing_fn=preprocessing_fn,
            framework='numpy')
        
        return evaluate_loss_accuracy(net, ds_test, loss_fn, device)
    
    
    def evaluate_train_metrics(net, loss_fn):
        
        ds_train_one_epoch = datasets.resisc45(
            split_type='train', epochs=1, 
            batch_size=args.bs, 
            dataset_dir=args.dataroot,
            preprocessing_fn=preprocessing_fn,
            framework='numpy')
        
        return evaluate_loss_accuracy(net, ds_train_one_epoch, loss_fn, device)
        
    
    def evaluate_adv_metrics(net, loss_fn):
    
        ds_adv = adversarial_datasets.resisc45_adversarial_224x224(
            split_type="adversarial", epochs=1,
            batch_size=args.bs,
            dataset_dir=args.dataroot,
            preprocessing_fn=preprocessing_fn,
            cache_dataset=True,
            framework="numpy",
            clean_key="clean",
            adversarial_key="adversarial_univpatch",
            targeted=False)
        
        return evaluate_loss_accuracy(net, ds_adv, loss_fn, device)
    
    writer = SummaryWriter(save_path, flush_secs=30)
    
    print("=> creating model '{}'".format(args.arch))
    model_kwargs = {"pretrained" : args.pretrained, "progress" : False}
    net = resisc_densenet121(model_kwargs).to(device)

    # Prepare training procedure
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-07, amsgrad=False, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_fn = nn.CrossEntropyLoss() # softmax cross entropy

    if args.fp16:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O3')

    #save_checkpoint(net, 100, 100, 0, save_path, ckpt_name)
    #save_amp_checkpoint(net, amp, optimizer, 100, 100, 0, save_path, ckpt_name)

    # Optionally resume from existing checkpoint
    if args.resume:
        if osp.isfile(args.resume):
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            start_epoch = checkpoint['epoch'] + 1
            torch.set_rng_state(checkpoint['rng_state'])
            global_step = start_epoch * (len(trainset) // args.bs)
    else:
        start_epoch = 0
        global_step = 0

    epoch = 0
    
    # begin main loop
    for batch, (x, y) in enumerate(ds_train):
        
        # evaluate performance metrics once per epoch at the beginning of every epoch
        if batch % (len(ds_train) // args.epochs) == 0:
            
            eval_start_time = time.time()
            net.eval()
            val_acc, val_loss = evaluate_test_metrics(net, loss_fn)
            print('Epoch [%d/%d], val acc %.4f, val loss %.4f, took %.2f sec, ' % (
                (epoch, args.epochs, val_acc, val_loss, time.time() - eval_start_time)))
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)
            
            epoch += 1
            
        # evaluate train metrics less often (every other epoch)
        if batch % (len(ds_train) // (args.epochs // 2)) == 0:
            
            eval_start_time = time.time()
            net.eval()
            trn_acc, trn_loss = evaluate_train_metrics(net, loss_fn)
            print('Epoch [%d/%d], trn acc %.4f, trn loss %.4f, took %.2f sec, ' % (
                (epoch, args.epochs, trn_acc, trn_loss, time.time() - eval_start_time)))
            writer.add_scalar('Loss/train', trn_loss, epoch)
            writer.add_scalar('Acc/train', trn_acc, epoch)
        
        # end per-epoch statistics
        net.train()
        
        #lr = adjust_learning_rate(optimizer, epoch, args.drop, args.lr)

        optimizer.zero_grad() # reset gradients
        
        x = torch.FloatTensor(x).to(device)
        y = torch.LongTensor(y).to(device)
        pred = net(x) # pre-softmax
        correct += (pred.argmax(dim=1) == y).sum()
        total += len(y)
        batch_loss = loss_fn(pred, y)
        #train_loss += batch_loss.item()

        if args.fp16:
            with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            batch_loss.backward() # bprop

        optimizer.step() # update parameters

        if batch % 10 == 0:
            print('Batch [{}/{}], train loss: {:.4f}'
                  .format(batch, len(ds_train) // args.epochs, batch_loss.item()))  #, train IoU: {:.4f}'
            writer.add_scalar('Loss/train mini-batch', batch_loss.item(), global_step)

            with torch.no_grad():
                for n, p in net.named_parameters():
                    if 'conv' in n.split('.'):
                        writer.add_scalar('L2norm/' + n, p.norm(2), global_step)
                    #elif 'scale' in n.split('.'):
                    #    writer.add_scalar('scale/' + n, p.item(), global_step)
                    # add scale here
        global_step += 1
        # epoch_time = time.time() - epoch_start_time

    '''
    images = inputs[:16].permute(0, 2, 3, 1) * c + c
    images = images.permute(0, 3, 1, 2)

    img_grid = torchvision.utils.make_grid(images)
    sig_grid = torchvision.utils.make_grid(sig(pred[:16]))
    lab_grid = torchvision.utils.make_grid(targets[:16].unsqueeze(dim=1).float())
    writer.add_image('images', img_grid, epoch)
    writer.add_image('predictions', sig_grid, epoch)
    writer.add_image('labels', lab_grid, epoch)
    '''

    if epoch % 10 == 0:
        #if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        if args.fp16:
            save_amp_checkpoint(net, amp, optimizer, val_loss, train_loss, epoch, save_path, ckpt_name)
        else:
            save_checkpoint(net, val_loss, train_loss, epoch, save_path, ckpt_name)

    print('Epoch [{}/{}], train loss: {:.4f}, val loss: {:.4f}, took {:.2f} s'
          .format(epoch + 1, args.epochs, train_loss, val_loss, epoch_time))

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            [epoch, args.lr, np.round(trn_loss, 4), np.round(val_loss, 4)])
    
    if args.fp16:
        save_amp_checkpoint(net, amp, optimizer, val_loss, trn_loss, epoch, save_path, ckpt_name)
    else:
        save_checkpoint(net, val_loss, trn_loss, epoch, save_path, ckpt_name)

    writer.close()
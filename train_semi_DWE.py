from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
from torch.backends import cudnn
import random
from PIL import Image
import sys
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset_2d import imagefloder_iitnn
from config.warmup_config.warmup import GradualWarmupScheduler
from config.train_test_config.train_test_config import print_train_loss_DWE, print_val_loss, print_train_eval_DWE, print_val_eval, save_val_best_2d, print_best
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='./checkpoints/semi_DWE')
    parser.add_argument('--path_seg_results', default='./seg_pred/semi_DWE')
    parser.add_argument('-pd', '--path_dataset', default='./GlaS')
    parser.add_argument('--dataset_name', default='GlaS', help='GlaS, EM, Ultrasound-Nerve, ROSSA')
    parser.add_argument('--input1', default='L')
    parser.add_argument('--input2', default='H')
    parser.add_argument('--sup_mark', default='20')
    parser.add_argument('--unsup_mark', default='80')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=1, type=float)
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-n', '--network', default='DWE', type=str)
    args = parser.parse_args()

    init_seeds(42)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # trained model save
    path_trained_models = args.path_trained_models + '/' + str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_trained_models):
        os.mkdir(path_trained_models)
    path_trained_models = path_trained_models+'/'+str(args.network)+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-cw='+str(args.unsup_weight)+'-w='+str(args.warm_up_duration)+'-'+ str(args.sup_mark)+'-'+str(args.unsup_mark)+'-'+str(args.input1)+'-'+str(args.input2)
    if not os.path.exists(path_trained_models):
        os.mkdir(path_trained_models)

    # seg results save
    path_seg_results = args.path_seg_results + '/' + str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_seg_results):
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results+'/'+str(args.network)+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-cw='+str(args.unsup_weight)+'-w='+str(args.warm_up_duration)+'-'+ str(args.sup_mark)+'-'+str(args.unsup_mark)+'-'+str(args.input1)+'-'+str(args.input2)
    if not os.path.exists(path_seg_results):
        os.mkdir(path_seg_results)

    if args.input1 == 'image':
        input1_mean = 'MEAN'
        input1_std = 'STD'
    else:
        input1_mean = 'MEAN_DB2_' + args.input1
        input1_std = 'STD_DB2_' + args.input1

    if args.input2 == 'image':
        input2_mean = 'MEAN'
        input2_std = 'STD'
    else:
        input2_mean = 'MEAN_DB2_' + args.input2
        input2_std = 'STD_DB2_' + args.input2

    data_transforms = data_transform_2d()
    data_normalize_1 = data_normalize_2d(cfg[input1_mean], cfg[input1_std])
    data_normalize_2 = data_normalize_2d(cfg[input2_mean], cfg[input2_std])

    dataset_train_unsup = imagefloder_iitnn(
        data_dir=args.path_dataset + '/train_unsup_'+args.unsup_mark,
        input1=args.input1,
        input2=args.input2,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize_1,
        data_normalize_2=data_normalize_2,
        sup=False,
        num_images=None,
    )
    num_images_unsup = len(dataset_train_unsup)

    dataset_train_sup = imagefloder_iitnn(
        data_dir=args.path_dataset + '/train_sup_'+args.sup_mark,
        input1=args.input1,
        input2=args.input2,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize_1,
        data_normalize_2=data_normalize_2,
        sup=True,
        num_images=num_images_unsup,
    )
    dataset_val = imagefloder_iitnn(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        input2=args.input2,
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize_1,
        data_normalize_2=data_normalize_2,
        sup=True,
        num_images=None,
    )

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()

    criterion = segmentation_loss(args.loss, False).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10 ** args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)

    since = time.time()
    count_iter = 0

    best_model = model
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter-1) % args.display_iter == 0:
            begin_time = time.time()

        model.train()

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0
        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0

        unsup_weight = args.unsup_weight

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup_1 = unsup_index['image']
            img_train_unsup_1 = Variable(img_train_unsup_1.cuda(non_blocking=True))
            img_train_unsup_2 = unsup_index['image_2']
            img_train_unsup_2 = Variable(img_train_unsup_2.cuda(non_blocking=True))

            optimizer.zero_grad()
            pred_train_unsup1, pred_train_unsup2 = model(img_train_unsup_1, img_train_unsup_2)

            max_train1 = torch.max(pred_train_unsup1, dim=1)[1]
            max_train2 = torch.max(pred_train_unsup2, dim=1)[1]
            max_train1 = max_train1.long()
            max_train2 = max_train2.long()

            loss_train_unsup = criterion(pred_train_unsup2, max_train1) + criterion(pred_train_unsup1, max_train2)
            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)

            sup_index = next(dataset_train_sup)
            img_train_sup_1 = sup_index['image']
            img_train_sup_1 = Variable(img_train_sup_1.cuda(non_blocking=True))
            img_train_sup_2 = sup_index['image_2']
            img_train_sup_2 = Variable(img_train_sup_2.cuda(non_blocking=True))
            mask_train_sup = sup_index['mask']
            mask_train_sup = Variable(mask_train_sup.cuda(non_blocking=True))

            pred_train_sup1, pred_train_sup2 = model(img_train_sup_1, img_train_sup_2)

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    score_list_train2 = pred_train_sup2
                    mask_list_train = mask_train_sup
                # else:
                elif 0 < i <= num_batches['train_sup'] / 32:
                    score_list_train1 = torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    score_list_train2 = torch.cat((score_list_train2, pred_train_sup2), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup1 = criterion(pred_train_sup1, mask_train_sup)
            loss_train_sup2 = criterion(pred_train_sup2, mask_train_sup)

            loss_train_sup = loss_train_sup1 + loss_train_sup2
            loss_train_sup.backward()
            optimizer.step()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss += loss_train.item()

        scheduler_warmup.step()

        if count_iter % args.display_iter == 0:

            torch.cuda.empty_cache()
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch+1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_cps, train_epoch_loss = print_train_loss_DWE(train_loss_sup_1, train_loss_sup_2, train_loss_unsup, train_loss, num_batches, print_num, print_num_half)
            train_eval_list1, train_eval_list2, train_m_jc1, train_m_jc2 = print_train_eval_DWE(cfg['NUM_CLASSES'], score_list_train1, score_list_train2, mask_list_train, print_num_half)

            with torch.no_grad():
                model.eval()

                for i, data in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val'] / 16:

                    inputs_val_1 = Variable(data['image'].cuda(non_blocking=True))
                    inputs_val_2 = Variable(data['image_2'].cuda(non_blocking=True))
                    mask_val = Variable(data['mask'].cuda(non_blocking=True))
                    name_val = data['ID']

                    optimizer.zero_grad()

                    outputs_val1, outputs_val2 = model(inputs_val_1, inputs_val_2)

                    if i == 0:
                        score_list_val1 = outputs_val1
                        score_list_val2 = outputs_val2
                        mask_list_val = mask_val
                        name_list_val = name_val
                    else:
                        score_list_val1 = torch.cat((score_list_val1, outputs_val1), dim=0)
                        score_list_val2 = torch.cat((score_list_val2, outputs_val2), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        name_list_val = np.append(name_list_val, name_val, axis=0)

                    loss_val_sup1 = criterion(outputs_val1, mask_val)
                    loss_val_sup2 = criterion(outputs_val2, mask_val)

                    val_loss_sup_1 += loss_val_sup1.item()
                    val_loss_sup_2 += loss_val_sup2.item()

                val_epoch_loss_sup1, val_epoch_loss_sup2 = print_val_loss(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half)
                val_eval_list1, val_eval_list2, val_m_jc1, val_m_jc2 = print_val_eval(cfg['NUM_CLASSES'], score_list_val1, score_list_val2, mask_list_val, print_num_half)
                best_val_eval_list, best_model, best_result = save_val_best_2d(cfg['NUM_CLASSES'], best_model, best_val_eval_list, best_result, model, model, score_list_val1, score_list_val2, name_list_val, val_eval_list1, val_eval_list2, path_trained_models, path_seg_results, cfg['PALETTE'])

                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time()-begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')

    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    print('=' * print_num)
    print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    print_best(cfg['NUM_CLASSES'], best_val_eval_list, best_model, best_result, path_trained_models, print_num_minus)
    print('=' * print_num)

from torchvision import transforms, datasets
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
from torch.backends import cudnn
import random
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from models.getnetwork import get_network
from dataload.dataset_2d import imagefloder_iitnn
from config.train_test_config.train_test_config import print_test_eval, save_test_2d
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
    parser.add_argument('-pd', '--path_dataset', default='./Ultrasound-Nerve')
    parser.add_argument('-p', '--path_model', default='./weights/best_Jc_0.6674.pth')
    parser.add_argument('--path_seg_results', default='./seg_pred/test')
    parser.add_argument('--dataset_name', default='Ultrasound-Nerve', help='LIDC-IDRI, Ultrasound-Nerve, GlaS')
    parser.add_argument('--input1', default='L')
    parser.add_argument('--input2', default='H')
    parser.add_argument('--if_mask', default=True)
    parser.add_argument('--threshold', default=0.5400, help='0.5600, 5400')
    parser.add_argument('--if_cct', default=False)
    parser.add_argument('--result', default='result2', help='result1, result2')
    parser.add_argument('-n', '--network', default='DWE')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    args = parser.parse_args()

    init_seeds(42)

    # Config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # Results Save
    if not os.path.exists(args.path_seg_results):
        os.mkdir(args.path_seg_results)
    path_seg_results = args.path_seg_results + '/' + str(dataset_name)
    if not os.path.exists(path_seg_results):
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results + '/' + str(os.path.splitext(os.path.split(args.path_model)[1])[0])
    if not os.path.exists(path_seg_results):
        os.mkdir(path_seg_results)

    # Dataset
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
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    num_batches = {'val': len(dataloaders['val'])}

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()

    state_dict = torch.load(args.path_model)
    model.load_state_dict(state_dict=state_dict)

    # Test
    since = time.time()

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(dataloaders['val']):

            inputs_test = Variable(data['image'].cuda(non_blocking=True))
            inputs_wavelet_test = Variable(data['image_2'].cuda(non_blocking=True))
            name_test = data['ID']
            if args.if_mask:
                mask_test = Variable(data['mask'].cuda(non_blocking=True))

            if args.if_cct:
                outputs_test1, outputs_test1_aux1, outputs_test1_aux2, outputs_test1_aux3, outputs_test2, outputs_test2_aux1, outputs_test2_aux2, outputs_test2_aux3 = model(inputs_test, inputs_wavelet_test)
            else:
                outputs_test1, outputs_test2 = model(inputs_test, inputs_wavelet_test)
            if args.result == 'result1':
                outputs_test = outputs_test1
            else:
                outputs_test = outputs_test2

            if args.if_mask:
                if i == 0:
                    score_list_test = outputs_test
                    name_list_test = name_test
                    mask_list_test = mask_test
                else:
                # elif 0 < i <= num_batches['val'] / 16:
                    score_list_test = torch.cat((score_list_test, outputs_test), dim=0)
                    name_list_test = np.append(name_list_test, name_test, axis=0)
                    mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
                torch.cuda.empty_cache()
            else:
                save_test_2d(cfg['NUM_CLASSES'], outputs_test, name_test, args.threshold, path_seg_results, cfg['PALETTE'])
                torch.cuda.empty_cache()

        print('=' * print_num)
        test_eval_list = print_test_eval(cfg['NUM_CLASSES'], score_list_test, mask_list_test, print_num_minus)
        save_test_2d(cfg['NUM_CLASSES'], score_list_test, name_list_test, test_eval_list[0], path_seg_results, cfg['PALETTE'])
        torch.cuda.empty_cache()


    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    print('-' * print_num)
    print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('=' * print_num)
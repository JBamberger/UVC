from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import cv2
import imageio

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import libs.model as video3d

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

import libs.VIP_test3_2videos_dense_fix as vip
#from models.videos.contrastSoftmaxLoss import ContrastSoftmaxLoss

# tps model
#from model.cnn_geometric_model import CNNGeometric, TwoStageCNNGeometric, FeatureCorrelation, featureL2Norm
#from model.loss import TransformedGridLoss, WeakInlierCount, TwoStageWeakInlierCount

#from geotnf.transformation import SynthPairTnf,SynthTwoPairTnf,SynthTwoStageTwoPairTnf
#from geotnf.transformation import GeometricTnf

from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict
from utils.torch_util import expand_dim


params = {}
#params['filelist'] = '/nfs.yoda/xiaolonw/pytorch_project2/someCycle/preprocess/VIP/vallist.txt'
params['filelist'] = 'vallist_vid.txt'
# params['batchSize'] = 24
params['imgSize'] = 240
params['cropSize'] = 240
params['cropSize2'] = 80
params['videoLen'] = 8
params['offset'] = 0
params['sideEdge'] = 80
params['predFrames'] = 1



def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=2e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/scratch/xiaolonw/pytorch_checkpoints/unsup3dnl_single_contrast', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--predDistance', default=0, type=int,
                    help='predict how many frames away')
parser.add_argument('--seperate2d', type=int, default=0, help='manual seed')
parser.add_argument('--batchSize', default=1, type=int,
                    help='batchSize')
parser.add_argument('--T', default=1.0, type=float,
                    help='temperature')
parser.add_argument('--gridSize', default=9, type=int,
                    help='temperature')
parser.add_argument('--classNum', default=49, type=int,
                    help='temperature')
parser.add_argument('--lamda', default=0.1, type=float,
                    help='temperature')
parser.add_argument('--use_softmax', type=str_to_bool, nargs='?', const=True, default=True,
                    help='pretrained_imagenet')
parser.add_argument('--use_l2norm', type=str_to_bool, nargs='?', const=True, default=False,
                    help='pretrained_imagenet')
parser.add_argument('--pretrained_imagenet', type=str_to_bool, nargs='?', const=True, default=False,
                    help='pretrained_imagenet')
parser.add_argument('--topk_vis', default=1, type=int,
                    help='topk_vis')

parser.add_argument('--videoLen', default=8, type=int,
                    help='predict how many frames away')
parser.add_argument('--frame_gap', default=2, type=int,
                    help='predict how many frames away')

parser.add_argument('--cropSize', default=240, type=int,
                    help='predict how many frames away')
parser.add_argument('--cropSize2', default=80, type=int,
                    help='predict how many frames away')
parser.add_argument('--temporal_out', default=4, type=int,
                    help='predict how many frames away')

parser.add_argument('--save_path', default='', type=str)
parser.add_argument('--resnet', action='store_true',
                    help='test on resnet')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

params['predDistance'] = state['predDistance']
print(params['predDistance'])

params['batchSize'] = state['batchSize']
print('batchSize: ' + str(params['batchSize']) )

print('temperature: ' + str(state['T']))

params['gridSize'] = state['gridSize']
print('gridSize: ' + str(params['gridSize']) )

params['classNum'] = state['classNum']
print('classNum: ' + str(params['classNum']) )

params['videoLen'] = state['videoLen']
print('videoLen: ' + str(params['videoLen']) )

params['cropSize'] = state['cropSize']
print('cropSize: ' + str(params['cropSize']) )
params['imgSize'] = state['cropSize']


params['cropSize2'] = state['cropSize2']
print('cropSize2: ' + str(params['cropSize2']) )
params['sideEdge'] = state['cropSize2']


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

print(args.gpu_id)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_loss = 0  # best test accuracy


def main():
    global best_loss
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    #if not os.path.isdir(args.checkpoint):
    #    mkdir_p(args.checkpoint)

    val_loader = torch.utils.data.DataLoader(
        vip.VIPSet(params, is_train=False, resnet=args.resnet),
        batch_size=int(params['batchSize']), shuffle=False,
        num_workers=args.workers, pin_memory=True)


    model = video3d.CycleTime(class_num=params['classNum'], trans_param_num=3,
                              pretrained=args.resnet, temporal_out=args.temporal_out,
                              use_resnet=args.resnet)
    # contrastnet = video3d.ContrastNetAll(class_num=params['classNum'])

    model = torch.nn.DataParallel(model).cuda()
    # contrastnet = torch.nn.DataParallel(contrastnet).cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # from IPython import embed; embed()
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999), weight_decay=0)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    title = 'videonet'
    """
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # contrastnet.load_state_dict(checkpoint['contrast_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Contrast Loss'])

        del checkpoint

    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Contrast Loss'])
    """


    if args.evaluate:
        print('\nEvaluation only')
        test_loss = test(val_loader, model, criterion, 1, use_cuda)
        print(' Test Loss:  %.8f' % (test_loss))
        return



palette_list = 'palette.txt'
f = open(palette_list, 'r')
palette = np.zeros((256, 3))
cnt = 0
for line in f:
    rows = line.split()
    palette[cnt][0] = int(rows[0])
    palette[cnt][1] = int(rows[1])
    palette[cnt][2] = int(rows[2])
    cnt = cnt + 1

f.close()
palette = palette.astype(np.uint8)




def test(val_loader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    save_objs = args.evaluate

    import os
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)# /scratch/xiaolonw/davis_results_mask_mixfcn/')
    # save_path = '/scratch/xiaolonw/davis_results_mask_mixfcn/'
    save_path = args.save_path + '/'
    # img_path  = '/scratch/xiaolonw/vlog_frames/'
    save_file = '%s/list.txt' % save_path

    fileout = open(save_file, 'w')

    end = time.time()

    # bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (imgs_total, lbls, meta) in enumerate(val_loader):


        finput_num_ori = params['videoLen']
        finput_num     = finput_num_ori

        # measure data loading time
        data_time.update(time.time() - end)
        imgs_total = torch.autograd.Variable(imgs_total.cuda())
        # patch2_total = torch.autograd.Variable(patch2_total.cuda())

        t00 = time.time()

        bs = imgs_total.size(0)
        total_frame_num = imgs_total.size(1)
        channel_num = imgs_total.size(2)
        height_len  = imgs_total.size(3)
        width_len   = imgs_total.size(4)

        assert(bs == 1)

        folder_paths = meta['folder_path'][0]
        framelist = os.listdir(folder_paths)
        foldernames = folder_paths.split('/')
        videoname = foldernames[-1]

        framelist = np.sort(framelist)

        temp_img_name = folder_paths + '/' + framelist[0]
        temp_img = cv2.imread(temp_img_name)

        real_height = temp_img.shape[0]
        real_width  = temp_img.shape[1]

        txt_path = folder_paths.replace('Images', 'Annotations/Instance_ids')
        all_files = os.listdir(txt_path)
        txt_files = []
        for i in range(len(all_files)):
            if 'txt' in all_files[i]:
                txt_files.append(all_files[i])
        txt_files = np.sort(txt_files)
        txt_file  = txt_path + '/' + txt_files[0]

        # f = open(txt_file, 'r')
        # class_corres = []
        # for line in f:
        #     rows = line.split()
        #     nowid = int(rows[0])
        #     nowlbl = int(rows[1])
        #     class_corres.append([nowid, nowlbl])
        # f.close()


        gridx = int(meta['gridx'].data.cpu().numpy()[0])
        gridy = int(meta['gridy'].data.cpu().numpy()[0])
        print('gridx: ' + str(gridx) + ' gridy: ' + str(gridy))
        print('total_frame_num: ' + str(total_frame_num))

        height_dim = int(params['cropSize'] / 8)
        width_dim  = int(params['cropSize'] / 8)

        # processing labels
        lbls = lbls[0].data.cpu().numpy()
        resize_lbl = cv2.resize(lbls[0], (params['cropSize'], params['cropSize']), interpolation=cv2.INTER_NEAREST)
        print(resize_lbl.shape)

        human_id_file = txt_file.replace('.txt', '.png')
        human_id_file = human_id_file.replace('Instance_ids', 'Human_ids')
        id_lbl = cv2.imread(human_id_file, 0)
        resize_id_lbl = cv2.resize(id_lbl, (params['cropSize'], params['cropSize']), interpolation=cv2.INTER_NEAREST)


        # lbl_set = palette[0 : 20, :]
        unique_lbls = np.unique(resize_id_lbl)
        lbl_set = palette[0 : 20, :]


        lbls_resize = np.zeros((1, resize_lbl.shape[0], resize_lbl.shape[1], len(lbl_set)))
        lbls_resize2 = np.zeros((1+params['videoLen'], height_dim, width_dim, len(lbl_set)))

        lbls_resize_id = np.zeros((1, resize_lbl.shape[0], resize_lbl.shape[1], len(unique_lbls)))
        lbls_resize2_id = np.zeros((1+params['videoLen'], height_dim, width_dim, len(unique_lbls)))

        # class labels


        for j in range(resize_lbl.shape[0]):
            for k in range(resize_lbl.shape[1]):

                pixellbl = resize_lbl[j, k].astype(np.uint8)
                lbls_resize[0, j, k, pixellbl] = 1

        lbls_resize2[0] = cv2.resize(lbls_resize[0], (height_dim, width_dim))

        for i in range(lbls_resize2.shape[0] - 1):
            lbls_resize2[i + 1] = lbls_resize2[0].copy()

        # instance labels
        for j in range(resize_id_lbl.shape[0]):
            for k in range(resize_id_lbl.shape[1]):

                pixellbl = resize_id_lbl[j, k].astype(np.uint8)
                new_pixelbl = -1
                for t in range(len(unique_lbls)):
                    if unique_lbls[t] == pixellbl:
                        new_pixelbl = t
                        break
                lbls_resize_id[0, j, k, new_pixelbl] = 1

        lbls_resize2_id[0] = cv2.resize(lbls_resize_id[0], (height_dim, width_dim))

        for i in range(lbls_resize2.shape[0] - 1):
            lbls_resize2[i + 1] = lbls_resize2[0].copy()
            lbls_resize2_id[i + 1] = lbls_resize2_id[0].copy()

        t02 = time.time()

        # print the images

        imgs_set = imgs_total.data
        imgs_set = imgs_set.cpu().numpy()
        imgs_set = imgs_set[0]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        imgs_toprint = []

        # ref image
        for t in range(imgs_set.shape[0]):
            img_now = imgs_set[t]

            for c in range(3):
                img_now[c] = img_now[c] * std[c]
                img_now[c] = img_now[c] + mean[c]

            img_now = img_now * 255
            img_now = np.transpose(img_now, (1, 2, 0))
            img_now = cv2.resize(img_now, (img_now.shape[0] * 2, img_now.shape[1] * 2) )

            imgs_toprint.append(img_now)

            imname  = save_path + str(batch_idx) + '_' + str(t) + '_frame.jpg'
            scipy.misc.imsave(imname, img_now)

        # for t in range(finput_num_ori):
        #
        #     nowlbl = lbls_new[t]
        #     imname  = save_path + str(batch_idx) + '_' + str(t) + '_label.jpg'
        #     scipy.misc.imsave(imname, nowlbl)


        now_batch_size = 1

        imgs_stack = []
        patch2_stack = []

        im_num = total_frame_num - finput_num_ori

        # for iter in range(im_num):
        #
        #     imgs = imgs_total[:, iter: iter + finput_num_ori, :, :, :]
        #     imgs2 = imgs_total[:, 0: finput_num_ori, :, :, :]
        #     imgs = torch.cat((imgs2, imgs), dim=1)
        #     patch2 = patch2_total[0, iter + finput_num_ori] # .unsqueeze(1)
        #
        #     imgs_stack.append(imgs)
        #     patch2_stack.append(patch2)
        #
        trans_out_2_set = []
        corrfeat2_set = []

        imgs_tensor = torch.Tensor(now_batch_size, finput_num, 3, params['cropSize'], params['cropSize'])
        target_tensor = torch.Tensor(now_batch_size, 1, 3, params['cropSize'], params['cropSize'])
        # patch2_tensor = torch.Tensor(now_batch_size, gridy * gridx, 3, params['cropSize2'], params['cropSize2'])
        imgs_tensor = torch.autograd.Variable(imgs_tensor.cuda())
        target_tensor = torch.autograd.Variable(target_tensor.cuda())
        # patch2_tensor = torch.autograd.Variable(patch2_tensor.cuda())


        t03 = time.time()

        result_path = save_path + '/results/' + videoname + '/'

        if os.path.exists(result_path) is False:
            os.makedirs(result_path )

        for iter in range(0, im_num, now_batch_size):

            print(iter)

            startid = iter
            endid   = iter + now_batch_size

            if endid > im_num:
                endid = im_num

            now_batch_size2 = endid - startid

            for i in range(now_batch_size2):

                imgs = imgs_total[:, iter + i + 1: iter + i + finput_num_ori, :, :, :]
                imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
                imgs = torch.cat((imgs2, imgs), dim=1)

                imgs_tensor[i] = imgs
                target_tensor[i, 0] = imgs_total[0, iter + i + finput_num_ori]

            corrfeat2_now = model(imgs_tensor, target_tensor)
            corrfeat2_now = corrfeat2_now.view(now_batch_size, finput_num_ori, corrfeat2_now.size(1), corrfeat2_now.size(2), corrfeat2_now.size(3))

            #for i in range(now_batch_size2):
            #    corrfeat2_set.append(corrfeat2_now[i].data.cpu().numpy())

        #t04 = time.time()
        #print(t04-t03, 'model forward', t03-t02, 'image prep')

        #for iter in range(total_frame_num - finput_num_ori):

            #if iter % 10 == 0:
            #    print(iter)

            imgs = imgs_total[:, iter + 1: iter + finput_num_ori, :, :, :]
            imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
            imgs = torch.cat((imgs2, imgs), dim=1)

            # patch2 = patch2_total[0, iter + finput_num_ori].unsqueeze(1)
            # patchnum = patch2.size(0)
            # print(patch2.size())

            # trans_out_2, corrfeat2 = model(imgs, patch2)
            #corrfeat2   = corrfeat2_set[iter]
            #corrfeat2   = torch.from_numpy(corrfeat2)
            corrfeat2 = corrfeat2_now[0].cpu()

            # bs * patchnum * self.temporal_out, 6
            # print(trans_out_2.size())
            # bs, patchnum, self.temporal_out, self.spatial_out1, self.spatial_out1, self.spatial_out2, self.spatial_out2
            # print(corrfeat2.size())

            out_frame_num = int(finput_num)
            height_dim = corrfeat2.size(2)
            width_dim = corrfeat2.size(3)

            corrfeat2 = corrfeat2.view(corrfeat2.size(0), height_dim, width_dim, height_dim, width_dim)
            corrfeat2 = corrfeat2.data.cpu().numpy()


            topk_vis = args.topk_vis
            vis_ids_h = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)
            vis_ids_w = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)

            t05 = time.time()

            atten1d  = corrfeat2.reshape(corrfeat2.shape[0], height_dim * width_dim, height_dim, width_dim)
            ids = np.argpartition(atten1d, -topk_vis, axis=1)[:, -topk_vis:]
            # ids = np.argsort(atten1d, axis=1)[:, -topk_vis:]

            hid = ids / width_dim
            wid = ids % width_dim

            vis_ids_h = hid.transpose(0, 2, 3, 1)
            vis_ids_w = wid.transpose(0, 2, 3, 1)

            t06 = time.time()

            img_now = imgs_toprint[iter + finput_num_ori]

            predlbls = np.zeros((height_dim, width_dim, len(lbl_set)))
            predlbls_id = np.zeros((height_dim, width_dim, len(unique_lbls)))

            # predlbls2 = np.zeros((height_dim * width_dim, len(lbl_set)))

            for t in range(finput_num):

                tt1 = time.time()

                h, w, k = np.meshgrid(np.arange(height_dim), np.arange(width_dim), np.arange(topk_vis), indexing='ij')
                h, w = h.flatten(), w.flatten()

                hh, ww = vis_ids_h[t].flatten(), vis_ids_w[t].flatten()
                hh = np.array(hh, np.uint8)
                ww = np.array(ww, np.uint8)

                #if t == 0:
                #    lbl = lbls_resize2[0, hh, ww, :]
                #else:
                #    lbl = lbls_resize2[t + iter, hh, ww, :]

                lbl = lbls_resize2[t, hh, ww, :]
                np.add.at(predlbls, (h, w), lbl * corrfeat2[t, hh, ww, h, w][:, None])

                #if t == 0:
                #    lbl = lbls_resize2_id[0, hh, ww, :]
                #else:
                #    lbl = lbls_resize2_id[t + iter, hh, ww, :]
                lbl = lbls_resize2_id[t, hh, ww, :]

                np.add.at(predlbls_id, (h, w), lbl * corrfeat2[t, hh, ww, h, w][:, None])


            t07 = time.time()
            # print(t07-t06, 'lbl proc', t06-t05, 'argsorts')

            predlbls = predlbls / finput_num
            predlbls_id = predlbls_id / finput_num

            # from IPython import embed; embed()

            now_scores = []
            cnt_scores = []

            for t in range(len(lbl_set)):
                nowt = t
                if np.sum(predlbls[:, :, nowt])  == 0:
                    continue

                predlbls[:, :, nowt] = predlbls[:, :, nowt] - predlbls[:, :, nowt].min()
                predlbls[:, :, nowt] = predlbls[:, :, nowt] / predlbls[:, :, nowt].max()

            for t in range(len(unique_lbls)):
                nowt = t
                if np.sum(predlbls_id[:, :, nowt])  == 0:
                    continue

                predlbls_id[:, :, nowt] = predlbls_id[:, :, nowt] - predlbls_id[:, :, nowt].min()
                predlbls_id[:, :, nowt] = predlbls_id[:, :, nowt] / predlbls_id[:, :, nowt].max()

            lbls_resize2[1] = predlbls
            lbls_resize2_id[1] = predlbls_id


            predlbls_cp = predlbls.copy()
            predlbls_cp = cv2.resize(predlbls_cp, (params['imgSize'], params['imgSize']))

            predlbls_cp_id = predlbls_id.copy()
            predlbls_cp_id = cv2.resize(predlbls_cp_id, (params['imgSize'], params['imgSize']))

            predlbls_val = np.zeros((params['imgSize'], params['imgSize'], 3))

            # ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)
            pred_mask = np.argmax(predlbls_cp, axis=-1)
            pred_mask_id = np.argmax(predlbls_cp_id, axis=-1)
            pred_mask_out = np.zeros((params['imgSize'], params['imgSize']))

            cnt_lbl = 0
            pair_vals = []
            indexes = np.zeros((len(lbl_set), len(unique_lbls)))
            now_scores = np.zeros((len(lbl_set), len(unique_lbls)))
            cnt_scores = np.zeros((len(lbl_set), len(unique_lbls)))

            for h in range(pred_mask.shape[0]):
                for w in range(pred_mask.shape[1]):
                    pixel_val = int(pred_mask[h, w])
                    pixel_id  = int(pred_mask_id[h, w])
                    if pixel_val == 0 or pixel_id == 0:
                        continue
                    if cnt_scores[pixel_val, pixel_id] == 0:
                        pair_vals.append([pixel_val, pixel_id])
                        cnt_lbl = cnt_lbl + 1
                        indexes[pixel_val, pixel_id] = cnt_lbl

                    cnt_scores[pixel_val, pixel_id] += 1
                    now_scores[pixel_val, pixel_id] += predlbls_cp[h, w, pixel_val] * predlbls_cp_id[h, w, pixel_id]

                    pred_mask_out[h, w] = indexes[pixel_val, pixel_id]


            print_lbl_set = palette[0 : cnt_lbl + 1, :]
            #pred_mask_color = np.array(print_lbl_set)[pred_mask_out.astype(np.int32)]
            #scipy.misc.imsave('pred_mask'+str(iter)+'.png', pred_mask_color)

            pred_mask_out = cv2.resize(pred_mask_out, (real_width, real_height), interpolation=cv2.INTER_NEAREST)

            predlbls_val = np.array(print_lbl_set)[pred_mask_out.astype(np.int32)]

            # predlbls_val = np.array(lbl_set)[np.argmax(predlbls_cp, axis=-1)]

            predlbls_val = predlbls_val.astype(np.uint8)
            predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[0], img_now.shape[1]), interpolation=cv2.INTER_NEAREST)

            img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

            imname  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_label.jpg'
            imname2  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_mask.png'

            eval_filename = framelist[iter]
            eval_filename = eval_filename.replace('.jpg', '.png')

            eval_result_path = result_path + eval_filename
            txt_result_path = eval_result_path.replace('.png', '.txt')

            f = open(txt_result_path, 'w')
            for t in range(len(pair_vals)):

                pixel_val = pair_vals[t][0]
                pixel_id  = pair_vals[t][1]
                avg_score = now_scores[pixel_val, pixel_id] / cnt_scores[pixel_val, pixel_id]

                outstr = str(pixel_val) + ' ' + str(avg_score) + '\n'
                f.write(outstr)
            f.close()

            scipy.misc.imsave(imname, np.uint8(img_with_heatmap))
            scipy.misc.imsave(imname2, np.uint8(predlbls_val))
            scipy.misc.imsave(eval_result_path, np.uint8(pred_mask_out))


    fileout.close()


        # plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=len(val_loader),
    #                 data=data_time.avg,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 )
    #     bar.next()
    # bar.finish()
    return losses.avg

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    epoch = state['epoch']
    filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # if is_best:
    #     shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

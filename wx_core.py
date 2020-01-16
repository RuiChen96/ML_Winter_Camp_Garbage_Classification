# -*- coding: UTF-8 -*-
from __future__ import print_function
import os
import json
import urllib
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from datasets.garbage_testset import im_loader
from tnn.network import net_utils

from wechatpy import create_reply
import wechatpy.messages as messages
from wechatpy.replies import ArticlesReply
from wechatpy.utils import ObjectDict

from inference.build_net import make_model
from args import args
import urllib.request
import seaborn as sns
from pylab import savefig
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

sns.set()

# ============
site_root = 'http://166.111.139.131'
ckpt = os.path.join('/home/ruichen/Documents/2019_autumn/model', 'model_29_9989_9366.pth')
inp_size = 256
gpu = 0

im_save_dir = '/home/ruichen/Documents/2019_autumn/data/garbage/wx/saved'
classes_name_file = 'garbage_names.txt'
fine_cls_urls_file = 'garbage_link.txt'

# ============

with open('garbage_classify_rule.json', 'r') as fp:
    cls_names = json.load(fp)
print(cls_names)

coarse_cls2ind = {"其他垃圾": 0, "厨余垃圾": 1, "可回收物": 2, "有害垃圾": 3}
class_names = []
im_urls = []
with open(classes_name_file, 'r') as f:
    for line in f.readlines():
        items = line.split(' ')
        if len(items) == 2:
            class_names.append(items[0].strip())
            im_urls.append(items[1].strip())
num_classes = len(class_names)
# print(class_names, im_urls)

fine_cls_urls = []
with open(fine_cls_urls_file, 'r') as fp:
    for line in fp.readlines():
        fine_cls_urls.append(line)

im_transform = transforms.Compose([
            transforms.CenterCrop(inp_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# load model
model_arch = 'resnext101_32x16d_wsl'

model = make_model(args)
# model = resnet50(num_classes=num_classes)
# net_utils.load_net(ckpt, model)
# checkpoint = torch.load(ckpt)
model.load_state_dict(torch.load(ckpt))
model = model.cuda(gpu)
model.eval()
print('load net from: {}'.format(ckpt))


def test_im(im_path):
    img = im_loader(im_path)
    img = im_transform(img)
    img_data = torch.unsqueeze(img, 0)

    im_var = Variable(img_data, volatile=True).cuda(gpu)
    output = model(im_var)
    output = torch.softmax(output, dim=1)
    # output, spatial_attn = model(im_var)
    # print('[log] spatial attn size: ', spatial_attn.size())
    print('[log] output:', output.data, output.size())
    predicts = output.data.squeeze().cpu().numpy()
    #
    # tmp_size = spatial_attn.size()
    # spatial_attn = spatial_attn.view(tmp_size[2], tmp_size[3])
    # spatial_attn = torch.reshape(spatial_attn, (spatial_attn.shape[2], spatial_attn.shape[3]))
    # print(spatial_attn.size())
    # spatial_attn = spatial_attn.detach().cpu().numpy()
    # fig = cv2.applyColorMap((spatial_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # cv2.imwrite('/home/ruichen/Documents/heatmap_cv2.png', fig)
    # vis_attn = normalize(vis_attn)
    # vis_attn = np.random.rand((128, 128))
    # sns_plot = sns.heatmap(vis_attn)
    # figure = sns_plot.get_figure()
    # figure.savefig('/home/ruichen/Documents/heatmap.png', dpi=400)
    # plt.close(figure)
    # print('[log] debug ---', vis_attn, vis_attn.shape)
    return predicts


def msg_handler(msg):
    if msg.type == 'image':
        im_id = msg.media_id
        im_url = msg.image
        save_name = os.path.join(im_save_dir, '{}.jpg'.format(im_id))
        urllib.request.urlretrieve(im_url, save_name)
        # print('download image from: {} \nto: {}'.format(im_url, save_name))
        print('[log] recive an image...')

        # resize
        im = cv2.imread(save_name)
        if im is None:
            reply = create_reply('图片上传失败', msg)
        else:
            min_size = min(im.shape[:2])
            scale = 300. / min_size
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            cv2.imwrite(save_name, im)
            print('[log] resize image: {}'.format(scale))

            pred = test_im(save_name)
            # print('[log] predictions: ', pred, 'sum: ', sum(pred))
            inds = np.arange(len(pred), dtype=np.int)
            inds = sorted(inds, key=lambda i: pred[i], reverse=True)
            print('[log] predict: ', end='')
            print(inds)

            garbages = []
            for ind in inds[:1]:
                pred_name = cls_names[str(ind)]
                coarse_cls = pred_name.split('/')[0]
                im_url_ind = coarse_cls2ind[coarse_cls]
                print('[log]: ind: ', ind, 'url: ', fine_cls_urls[ind])
                garbage = {
                    'title': '{}: {:.1f}%'.format(cls_names[str(ind)], pred[ind] * 100),
                    'description': '{}: {:.1f}% [点击查看详情]'.format(cls_names[str(ind)], pred[ind] * 100),
                    'image': im_urls[im_url_ind],
                    'url': fine_cls_urls[ind]
                }
                garbages.append(garbage)
            reply = create_reply(garbages, msg)
    else:
        text = '欢迎来到 Garbage 分类小助手！\n'
        text += '我们是 DuckDuckGo（冲鸭）队，请投我们一票！'
        text += '拿起手中的物品，拍照并发送照片，看看它们是什么种类的垃圾～'
        # for name in class_names:
        #     text += '{} '.format(name)
        # text += '{}余类垃圾'.format(len(cls_names))
        reply = create_reply(text, msg)
        # reply = create_reply('Sorry, can not handle this for now', msg)

    return reply

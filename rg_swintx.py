import os
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import collections
from mmaction.apis import init_recognizer
import torch.nn.functional as F
import json


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoData(Dataset):
    def __init__(self, root, clip_len, detect_path):
        self.root = root
        self.detect_path = detect_path
        self.clip_len = clip_len
        self.video_name = []
        for name in sorted(list(os.listdir(root))):
            if args.op == 'pool':
                if os.path.exists(os.path.join(pool_save, os.path.splitext(name)[0] + '.npy')):
                    continue
            elif args.op == 'orig':
                if os.path.exists(os.path.join(orig_save, os.path.splitext(name)[0] + '.npy')):
                    continue
            if args.action == 'all' or name.split('_')[0] == args.action:
                self.video_name.append(name)

        self.transform = transforms.Compose([
            transforms.Resize(256),         # 短边放缩到256
            transforms.CenterCrop((224, 300)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675/255, 116.28/255, 103.53/255], std=[58.395/255, 57.12/255, 57.375/255])
        ])

    def __getitem__(self, index):
        video_path = os.path.join(self.root, self.video_name[index])
        frames = self.read_video(video_path)
        num_clips = len(frames) // self.clip_len

        select_cnt = num_clips * self.clip_len
        '''
        1. 随机舍弃余数部分的帧
        '''
        # select_idx = np.random.choice(len(frames), select_cnt, replace=False).tolist()
        # select_idx.sort()
        # frames = frames[select_idx]
        '''
        2. 均匀舍弃头尾，取中部
        '''
        start = (len(frames) - select_cnt) // 2
        end = len(frames) - select_cnt - start
        frames = frames[start: len(frames) - end]

        # [T, C, H, W] => [N, T, C, H, W]
        frames = frames.view(-1, self.clip_len, 3, 224, 224)        # 按clip截开
        # [N, T, C, H, W] => [N, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)
        return frames, os.path.splitext(self.video_name[index])[0]

    def __len__(self):
        return len(self.video_name)

    def read_video(self, path):
        vid = cv2.VideoCapture(path)
        name = os.path.splitext(os.path.split(path)[-1])[0]
        frames = []
        id = 0

        # tmp = './test'
        # if not os.path.exists(tmp):
        #     os.makedirs(tmp)
        while True:
            ok, frame = vid.read()
            if not ok:
                break
            # if args.erase:
            #     with open(os.path.join(self.detect_path, name, '%05d' % id + '.json'), 'r') as f:
            #         info = json.loads(json.load(f))
            #     frame = self.erase(frame, info)
                # cv2.imwrite(os.path.join(tmp, '%05d' % id + '.png'), frame)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
            frame = self.transform(frame)    # (c, h, w)
            frames.append(frame)
            id += 1
        frames = torch.stack(frames, dim=0)     # (t, c, h, w)
        return frames

    def erase(self, img, detect_res):
        h, w = img.shape[0:2]
        max_area = 0
        max_coor = ()
        avg_color = np.mean(img, axis=(0, 1))
        for entry in detect_res:
            xmin, ymin, xmax, ymax = int(entry['xmin']), int(entry['ymin']), int(entry['xmax']), int(entry['ymax'])
            if entry['name'] != 'person' or (xmax - xmin) * (ymax - ymin) < max_area:
                continue
            max_area = (xmax - xmin) * (ymax - ymin)
            max_coor = (xmin, ymin, xmax, ymax)
        if max_area:
            xmin, ymin, xmax, ymax = max_coor
            xmin, ymin, xmax, ymax = max(0, xmin - 10), max(0, ymin - 10), min(w - 1, xmax + 10), min(h - 1, ymax + 10)
            img[ymin:ymax, xmin:xmax] = avg_color
        return img


def load_model(device):
    root = '../../Video-Swin-Transformer-master'
    config_file = os.path.join(root, 'configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py')
    # cpt = './swin_base_patch244_window877_kinetics600_22k.pth'
    cpt = os.path.join(root, 'swin_base_patch244_window877_kinetics600_22k.pth')

    device = 'cuda:' + str(device)  # or 'cpu'
    device = torch.device(device)

    model = init_recognizer(config_file, cpt, device=device)
    return model.backbone


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--gpu', type=str, default='0')
    parse.add_argument('--op', type=str, default='pool')
    parse.add_argument('--action', type=str, default='all')
    parse.add_argument('--erase', type=bool, default=False)
    args = parse.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    setup_seed(0)
    clip_len = 32

    data_path = '/home/share/angchi/datasets/RG_public/videos'
    detect_path = '/home/angchi/action assessment/rg_feat/detect/'

    orig_save = '/home/share/angchi/rg_feat/swintx_orig_fps25_clip{}'.format(clip_len)
    pool_save = '/home/angchi/action assessment/rg_feat/swintx_avg_fps25_clip{}'.format(clip_len)
    if not os.path.exists(orig_save):
        os.makedirs(orig_save)
    if not os.path.exists(pool_save):
        os.makedirs(pool_save)

    model = load_model(args.gpu)
    dataset = VideoData(data_path, clip_len, detect_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    with torch.no_grad():
        for i, (data, name) in enumerate(loader):
            print(name[0])
            data = data.to(int(args.gpu))
            data = data.squeeze(0)      # (N, C, T, H, W)
            orig_feat, avg_feat = [], []

            batch_size = 4         # 每次送多少clip
            for j in range((len(data) + batch_size-1) // batch_size):
                end = min(j * batch_size + batch_size, len(data))
                feat = model(data[j * batch_size:end])      # (B, C, T/2, H/32, H/32)

                if args.op == 'pool':
                    avg_feat.append(torch.flatten(F.adaptive_avg_pool3d(feat, (1, 1, 1)), start_dim=1).cpu().numpy())  # (B, C)
                elif args.op == 'orig':
                    orig_feat.append(feat.cpu().numpy())
                    # b, c, t, h, w = feat.shape
                    # feat = feat.transpose(1, 2).view(-1, c, h, w)    # (BT, C, H, W)
                    # orig_feat.append(feat.cpu().numpy())

                # import pdb
                # pdb.set_trace()
            if args.op == 'pool':
                np.save(os.path.join(pool_save, name[0] + '.npy'), np.concatenate(avg_feat, 0))
            elif args.op == 'orig':
                np.save(os.path.join(orig_save, name[0] + '.npy'), np.concatenate(orig_feat, 0))

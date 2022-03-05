from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F


class RGDataset(Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=26, action_type='Ball',
                 score_type='Total_Score', train=True):
        if action_type == 'all' and not train:
            raise SystemError
        self.train = train
        self.video_path = video_feat_path
        self.erase_path = video_feat_path + '_erTrue'

        self.clip_num = clip_num
        self.labels = self.read_label(label_path, score_type, action_type)

    def read_label(self, label_path, score_type, action_type):
        fr = open(label_path, 'r')
        idx = {'Difficulty_Score': 1, 'Execution_Score': 2, 'Total_Score': 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            if action_type == 'all' or action_type == line[0].split('_')[0]:
                labels.append([line[0], float(line[idx[score_type]])])
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, self.labels[idx][0] + '.npy'))

        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat

        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / 25

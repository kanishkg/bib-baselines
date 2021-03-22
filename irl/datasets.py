import json
import os
import math

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data

ACTION_LIST = [[0, 1], [1, 0], [1, 1],
               [0, -1], [-1, 0], [-1, -1],
               [0, 0], [-1, 1], [1, -1]]


#
# def collate_function(batch):
#     dem_states = torch.stack([item[0] for item in batch])
#     dem_actions = torch.stack([item[1] for item in batch])
#     test_states = torch.stack([item[2] for item in batch])
#     test_actions = torch.stack([item[3] for item in batch])
#     dem_l = torch.tensor([item[4] for item in batch])
#     test_l = [item[5] for item in batch]
#     return [dem_states, dem_actions, test_states, test_actions, dem_l, test_l]


class CacheDataset(torch.utils.data.Dataset):

    def __init__(self, path, types=None, size=None, mode="train", process_data=0):

        self.path = path
        self.types = types
        self.size = size
        self.mode = mode
        self.json_list = []
        self.path_list = []

        # read video files
        for t in types:
            print(f'reading files of type {t} in {mode}')
            self.path_list += [os.path.join(self.path, x) for x in os.listdir(self.path) if
                               x.endswith(f'{t}e.mp4')]
            self.json_list += [os.path.join(self.path, x) for x in os.listdir(self.path) if
                               x.endswith(f'{t}e.json')]

        self.path_list = sorted(self.path_list)
        self.json_list = sorted(self.json_list)

        self.data_tuples = []
        # split for train and val
        if mode == 'train':
            self.path_list = self.path_list[:int(0.8 * len(self.path_list))]
            self.json_list = self.json_list[:int(0.8 * len(self.json_list))]
        elif mode == 'val':
            self.path_list = self.path_list[int(0.8 * len(self.path_list)):]
            self.json_list = self.json_list[int(0.8 * len(self.json_list)):]
        elif mode == 'test':
            pass

        if process_data:

            print('processing files')
            for j, v in zip(self.json_list, self.path_list):
                print(j)
                with open(j, 'r') as f:
                    state = json.load(f)
                ep_lens = [len(x) for x in state]
                past_len = 0
                for e, l in enumerate(ep_lens):
                    self.data_tuples.append([])
                    # skip first 30 frames and last 83 frames
                    for f in range(30, l - 83 - 1):
                        # find action taken; this calculation is approximate
                        f0x, f0y = state[e][f]['agent'][0]
                        f1x, f1y = state[e][f + 1]['agent'][0]
                        dx = int((f1x - f0x) / 2)
                        dy = int((f1x - f0x) / 2)
                        action = ACTION_LIST.index([dx, dy])
                        self.data_tuples[-1].append((v, past_len + f, action))
                    print(len(self.data_tuples[-1]))
                    assert len(self.data_tuples[-1]) > 0
                    past_len += l

            index_dict = {'data_tuples': self.data_tuples}
            with open(os.path.join(self.path, f'index_bib_{mode}.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:
            with open(os.path.join(self.path, f'index_bib_{mode}.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_tuples = index_dict['data_tuples']

    def _get_frames(self, video, frames_idx):
        cap = cv2.VideoCapture(video)
        frames = []
        # read frames at ids and resize
        for i, f in enumerate(frames_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            _, frame = cap.read()
            if self.size is not None:
                assert frame is not None, f'frame is empty {f}, {video}, i'
                frame = cv2.resize(frame, self.size)
            frame = torch.tensor(frame).permute(2, 0, 1)
            frames.append(frame)

        # return frames as a torch tensor f x c x w x h
        frames = torch.stack(frames, dim=0)
        frames = frames.to(torch.float32) / 255.
        cap.release()
        return frames

    def __getitem__(self, idx):
        # works only with batch size of 1
        print(len(self.data_tuples[idx]))
        print(self.data_tuples[idx][0][0])
        video = self.data_tuples[idx][0][0]
        frames_idx = [d[1] for d in self.data_tuples[idx]]
        actions = torch.tensor([d[2] for d in self.data_tuples[idx]])
        frames = self._get_frames(video, frames_idx)
        return frames, actions

    def __len__(self):
        return len(self.data_tuples)


class BIBDataset(torch.utils.data.Dataset):

    def __init__(self, path, types=None, mode="train", num_trials=9, max_len=150):
        self.path = path
        self.types = types
        self.mode = mode
        self.num_trials = num_trials
        self.max_len = max_len

        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]
        self.h5file = h5py.File(f'{self.path}_{self.mode}.h5', 'r')
        self.tot_trials = int(len(self.h5file.keys()) / 2)

    def get_trial(self, trial):
        # retrieve state embeddings and actions from cached file
        with h5py.File(f'{self.path}_{self.mode}.h5', 'r') as f:
            states_h5 = self.h5file[f'{trial}_s']
            actions_h5 = self.h5file[f'{trial}_a']
        l, state_dim = states_h5.shape
        states = np.zeros((self.max_len, state_dim))
        states[:l, :] = states_h5
        actions = np.zeros((self.max_len))
        actions[:l] = actions_h5
        return states, actions, l

    def __getitem__(self, idx):
        # retrieve 2 expert trajectories
        ep_num = idx // self.ep_combs
        trial_d = ep_num * self.ep_combs + self.eps[idx % self.ep_combs][0]
        trial_t = ep_num * self.ep_combs + self.eps[idx % self.ep_combs][1]
        dem_states, dem_actions, dem_l = self.get_trial(trial_d)
        test_states, test_actions, test_l = self.get_trial(trial_t)
        test_mask = np.zeros((self.max_len))
        test_mask[:test_l] = 1
        return dem_states, dem_actions, test_states, test_actions, dem_l, test_l, test_mask

    def __len__(self):
        # totaltrials / 9 * 9p2 when num trials is 9
        return self.tot_trials * (self.num_trials - 1)

import json
import os
import random
import pickle

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data

ACTION_LIST = [[0, 1], [1, 0], [1, 1],
               [0, -1], [-1, 0], [-1, -1],
               [0, 0], [-1, 1], [1, -1]]


class CacheDataset(torch.utils.data.Dataset):

    def __init__(self, path, types=None, size=None, mode="train", process_data=0):

        self.path = path
        self.types = types
        self.size = size
        self.mode = mode
        self.json_list = []
        self.path_list = []

        # read video files
        type_str = '_'.join(types)

        for t in types:
            print(f'reading files of type {t} in {mode}')
            paths = [os.path.join(self.path, x) for x in os.listdir(self.path) if
                     x.endswith(f'{t}.mp4')]
            jsons = [os.path.join(self.path, x) for x in os.listdir(self.path) if
                     x.endswith(f'{t}.json') and 'index' not in x]

            paths = sorted(paths)
            jsons = sorted(jsons)

            if mode == 'train':
                self.path_list += paths[:int(0.8 * len(jsons))]
                self.json_list += jsons[:int(0.8 * len(jsons))]
            elif mode == 'val':
                self.path_list += paths[int(0.8 * len(jsons)):]
                self.json_list += jsons[int(0.8 * len(jsons)):]
            else:
                self.path_list += paths
                self.json_list += jsons

        self.data_tuples = []

        if process_data:

            print(f'processing files {len(self.json_list)}')
            for j, v in zip(self.json_list, self.path_list):
                print(j)
                with open(j, 'r') as f:
                    state = json.load(f)
                ep_lens = [len(x) for x in state]
                past_len = 0
                for e, l in enumerate(ep_lens):
                    self.data_tuples.append([])
                    # skip first 30 frames and last 83 frames
                    for f in range(30, l - 83):
                        # find action taken; this calculation is approximate
                        f0x, f0y = state[e][f]['agent'][0]
                        f1x, f1y = state[e][f + 1]['agent'][0]
                        dx = (f1x - f0x) / 2.
                        dy = (f1y - f0y) / 2.
                        action = [dx, dy]
                        # action = ACTION_LIST.index([dx, dy])
                        self.data_tuples[-1].append((v, past_len + f, action))
                    print(len(self.data_tuples[-1]))
                    assert len(self.data_tuples[-1]) > 0
                    past_len += l

            index_dict = {'data_tuples': self.data_tuples}
            with open(os.path.join(self.path, f'index_bib_{mode}_{type_str}.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:
            with open(os.path.join(self.path, f'index_bib_{mode}_{type_str}.json'), 'r') as fp:
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


class TransitionDataset(torch.utils.data.Dataset):

    def __init__(self, path, types=None, mode="train", num_context=30, num_test=1, num_trials=9, action_range=10):
        self.path = path
        self.types = types
        self.mode = mode
        self.num_trials = num_trials
        self.num_context = num_context
        self.num_test = num_test
        self.action_range = action_range
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]
        types_str = '_'.join(self.types)
        with open(f'{self.path}_{self.mode}_{types_str}.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
        self.tot_trials = len(self.data.keys()) // 2

    def get_trial(self, trials, num_transitions, step=1):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []
        for t in trials:
            trial_len += [(t, n) for n in range(0, len(self.data[f'{t}_s']), step)]
        random.shuffle(trial_len)
        if len(trial_len) < num_transitions:
            return None, None, False
        for t, n in trial_len[:num_transitions]:
            states.append(self.data[f'{t}_s'][n, :])
            actions_xy = self.data[f'{t}_a'][n:n+self.action_range, :]
            actions_xy = np.mean(actions_xy, axis=0)
            action = np.array(actions_xy)
            actions.append(action)
        states = torch.tensor(np.array(states)).double()
        actions = torch.tensor(np.array(actions)).double()
        return states, actions, True

    def __getitem__(self, idx):
        # retrieve 2 expert trajectories
        dem = False
        test = False
        while not dem or not test:
            ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]
            random.shuffle(ep_trials)
            dem_states, dem_actions, dem = self.get_trial(ep_trials[:-1], self.num_context)
            test_states, test_actions, test = self.get_trial([ep_trials[-1]], self.num_test, step=self.action_range)
        return dem_states, dem_actions, test_states, test_actions

    def __len__(self):
        return self.tot_trials // self.num_trials


class TestTransitionDataset(torch.utils.data.Dataset):

    def __init__(self, path, type=None, num_trials=9, action_range=10):
        self.path = path
        self.num_trials = num_trials
        self.action_range = action_range
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]

        # load expected and unexpected caches
        with open(f'{self.path}_test_{type}e.pickle', 'rb') as handle:
            self.data_expected = pickle.load(handle)
        with open(f'{self.path}_test_{type}u.pickle', 'rb') as handle:
            self.data_unexpected = pickle.load(handle)

        self.tot_trials = len(self.data_expected.keys()) // 2

    def get_trial(self, trials, data):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []
        for t in trials:
            trial_len += [(t, n) for n in range(len(data[f'{t}_s']))]
        for t, n in trial_len:
            states.append(data[f'{t}_s'][n, :])
            actions_xy = data[f'{t}_a'][n:n+self.action_range, :]
            actions_xy = np.mean(actions_xy, axis=0)
            action = np.array(actions_xy)
            actions.append(action)
        states = torch.tensor(np.array(states)).double()
        actions = torch.tensor(np.array(actions)).double()
        return states, actions

    def __getitem__(self, idx):
        # only works with batch size 1
        ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]
        random.shuffle(ep_trials)

        # retrieve complete fam trajectories
        fam_expected_states, fam_expected_actions = self.get_trial(ep_trials[:-1], self.data_expected)
        fam_unexpected_states, fam_unexpected_actions = self.get_trial(ep_trials[:-1], self.data_unexpected)

        # retrieve complete test trajectories
        test_expected_states, test_expected_actions = self.get_trial([ep_trials[-1]], self.data_expected)
        test_unexpected_states, test_unexpected_actions = self.get_trial([ep_trials[-1]], self.data_unexpected)

        return fam_expected_states, fam_expected_actions, test_expected_states, test_expected_actions, \
               fam_unexpected_states, fam_unexpected_actions, test_unexpected_states, test_unexpected_actions

    def __len__(self):
        return self.tot_trials // self.num_trials

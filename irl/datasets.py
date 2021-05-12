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
            if self.data[f'{t}_a'].shape[0] > n + self.action_range:
                actions_xy = self.data[f'{t}_a'][n:n + self.action_range, :]
            else:
                actions_xy = self.data[f'{t}_a'][n:, :]
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
            actions_xy = data[f'{t}_a'][n:n + self.action_range, :]
            actions_xy = np.mean(actions_xy, axis=0)
            action = np.array(actions_xy)
            actions.append(action)
        states = torch.tensor(np.array(states)).double()
        actions = torch.tensor(np.array(actions)).double()
        return states, actions

    def __getitem__(self, idx):
        # only works with batch size 1
        ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]

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


class RawTransitionDataset(torch.utils.data.Dataset):

    def __init__(self, path, types=None, size=None, mode="train", num_context=30, num_test=1, num_trials=9,
                 action_range=10,
                 process_data=0):
        self.path = path
        self.types = types
        self.size = size
        self.mode = mode
        self.num_trials = num_trials
        self.num_context = num_context
        self.num_test = num_test
        self.action_range = action_range
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]
        types_str = '_'.join(self.types)

        self.path_list = []
        self.json_list = []
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
            with open(os.path.join(self.path, f'index_bib_{mode}_{types_str}.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:
            with open(os.path.join(self.path, f'index_bib_{mode}_{types_str}.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_tuples = index_dict['data_tuples']

        self.tot_trials = len(self.path_list) * 9

    def _get_frame(self, video, frame_idx):
        cap = cv2.VideoCapture(video)
        # read frame at id and resize
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()
        if self.size is not None:
            assert frame is not None, f'frame is empty {frame_idx}, {video}'
            frame = cv2.resize(frame, self.size)
        frame = torch.tensor(frame).permute(2, 0, 1)
        # return frames as a torch tensor f x c x w x h
        frame = frame.to(torch.float32) / 255.
        cap.release()
        return frame

    def get_trial(self, trials, num_transitions, step=1):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []

        for t in trials:
            trial_len += [(t, n) for n in range(0, len(self.data_tuples[t]), step)]
        random.shuffle(trial_len)
        if len(trial_len) < num_transitions:
            return None, None, False

        for t, n in trial_len[:num_transitions]:
            video = self.data_tuples[t][n][0]
            states.append(self._get_frame(video, self.data_tuples[t][n][1]))

            if len(self.data_tuples[t]) > n + self.action_range:
                actions_xy = [d[2] for d in self.data_tuples[t][n:n + self.action_range]]
            else:
                actions_xy = [d[2] for d in self.data_tuples[t][n:]]
            actions_xy = np.array(actions_xy)
            actions_xy = np.mean(actions_xy, axis=0)
            actions.append(actions_xy)
        states = torch.stack(states, dim=0)
        actions = torch.tensor(np.array(actions))
        return states, actions, True

    def __getitem__(self, idx):
        # retrieve 2 expert trajectories
        dem = False
        test = False
        while not dem or not test:
            ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]
            random.shuffle(ep_trials)
            dem_states, dem_actions, dem = self.get_trial(ep_trials[:-1], self.num_context)
            test_states, test_actions, test = self.get_trial([ep_trials[-1]], self.num_test,
                                                             step=self.action_range)
        return dem_states, dem_actions, test_states, test_actions

    def __len__(self):
        return self.tot_trials // self.num_trials


class TestRawTransitionDataset(torch.utils.data.Dataset):

    def __init__(self, path, types=None, size=None, mode="train", num_context=30, num_test=1, num_trials=9,
                 action_range=10, process_data=0):
        self.path = path
        self.types = types
        self.size = size
        self.mode = mode
        self.num_trials = num_trials
        self.num_context = num_context
        self.num_test = num_test
        self.action_range = action_range
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]

        self.path_list_exp = []
        self.json_list_exp = []
        self.path_list_un = []
        self.json_list_un = []

        print(f'reading files of type {types} in {mode}')
        paths_expected = [os.path.join(self.path, x) for x in os.listdir(self.path) if
                 x.endswith(f'{types}e.mp4')]
        jsons_expected = [os.path.join(self.path, x) for x in os.listdir(self.path) if
                 x.endswith(f'{types}e.json') and 'index' not in x]
        paths_unexpected = [os.path.join(self.path, x) for x in os.listdir(self.path) if
                 x.endswith(f'{types}u.mp4')]
        jsons_unexpected = [os.path.join(self.path, x) for x in os.listdir(self.path) if
                 x.endswith(f'{types}u.json') and 'index' not in x]


        self.path_list_exp += paths_expected
        self.json_list_exp += jsons_expected
        self.path_list_un += paths_unexpected
        self.json_list_un += jsons_unexpected

        self.data_unexpected = []
        self.data_expected = []
        if process_data:

            print(f'processing files {len(self.json_list_exp)}')
            for j, v in zip(self.json_list_exp, self.path_list_exp):
                print(j)
                with open(j, 'r') as f:
                    state = json.load(f)
                ep_lens = [len(x) for x in state]
                past_len = 0
                for e, l in enumerate(ep_lens):
                    self.data_expected.append([])
                    # skip first 30 frames and last 83 frames
                    for f in range(30, l - 83):
                        # find action taken; this calculation is approximate
                        f0x, f0y = state[e][f]['agent'][0]
                        f1x, f1y = state[e][f + 1]['agent'][0]
                        dx = (f1x - f0x) / 2.
                        dy = (f1y - f0y) / 2.
                        action = [dx, dy]
                        # action = ACTION_LIST.index([dx, dy])
                        self.data_expected[-1].append((v, past_len + f, action))
                    print(len(self.data_expected[-1]))
                    assert len(self.data_expected[-1]) > 0
                    past_len += l

            index_dict = {'data_tuples': self.data_expected}
            with open(os.path.join(self.path, f'index_bib_test_{types}e.json'), 'w') as fp:
                json.dump(index_dict, fp)

            print(f'processing files {len(self.json_list_un)}')
            for j, v in zip(self.json_list_un, self.path_list_un):
                print(j)
                with open(j, 'r') as f:
                    state = json.load(f)
                ep_lens = [len(x) for x in state]
                past_len = 0
                for e, l in enumerate(ep_lens):
                    self.data_unexpected.append([])
                    # skip first 30 frames and last 83 frames
                    for f in range(30, l - 83):
                        # find action taken; this calculation is approximate
                        f0x, f0y = state[e][f]['agent'][0]
                        f1x, f1y = state[e][f + 1]['agent'][0]
                        dx = (f1x - f0x) / 2.
                        dy = (f1y - f0y) / 2.
                        action = [dx, dy]
                        # action = ACTION_LIST.index([dx, dy])
                        self.data_unexpected[-1].append((v, past_len + f, action))
                    print(len(self.data_unexpected[-1]))
                    assert len(self.data_unexpected[-1]) > 0
                    past_len += l

            index_dict = {'data_tuples': self.data_unexpected}
            with open(os.path.join(self.path, f'index_bib_test_{types}u.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:

            with open(os.path.join(self.path, f'index_bib_test_{types}e.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_expected = index_dict['data_tuples']
            with open(os.path.join(self.path, f'index_bib_test_{types}u.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_unexpected = index_dict['data_tuples']

    def _get_frame(self, video, frame_idx):
        cap = cv2.VideoCapture(video)
        # read frame at id and resize
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()
        if self.size is not None:
            assert frame is not None, f'frame is empty {frame_idx}, {video}'
            frame = cv2.resize(frame, self.size)
        frame = torch.tensor(frame).permute(2, 0, 1)
        # return frames as a torch tensor f x c x w x h
        frame = frame.to(torch.float32) / 255.
        cap.release()
        return frame

    def get_trial(self, trials, data, step=1, shuffle=False):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []

        for t in trials:
            trial_len += [(t, n) for n in range(0, len(data[t]), step)]
        if shuffle:
            random.shuffle(trial_len)
            trial_len = trial_len[:100]
        for t, n in trial_len:
            video = data[t][n][0]
            states.append(self._get_frame(video, data[t][n][1]))
            actions_xy = [d[2] for d in data[t][n:n + self.action_range]]
            actions_xy = np.array(actions_xy)
            actions_xy = np.mean(actions_xy, axis=0)
            actions.append(actions_xy)

        states = torch.stack(states, dim=0)
        actions = torch.tensor(np.array(actions))
        return states, actions

    def __getitem__(self, idx):
        # retrieve 2 expert trajectories
        ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]

        # retrieve complete fam trajectories
        fam_expected_states, fam_expected_actions = self.get_trial(ep_trials[:-1], self.data_expected, shuffle=True)
        fam_unexpected_states, fam_unexpected_actions = self.get_trial(ep_trials[:-1], self.data_unexpected,
                                                                       shuffle=True)

        # retrieve complete test trajectories
        test_expected_states, test_expected_actions = self.get_trial([ep_trials[-1]], self.data_expected,
                                                                     step=self.action_range)
        test_unexpected_states, test_unexpected_actions = self.get_trial([ep_trials[-1]], self.data_unexpected,
                                                                         step=self.action_range)

        return fam_expected_states, fam_expected_actions, test_expected_states, test_expected_actions, \
               fam_unexpected_states, fam_unexpected_actions, test_unexpected_states, test_unexpected_actions

    def __len__(self):
        return len(self.data_expected) // 9


class RewardTransitionDataset(torch.utils.data.Dataset):

    def __init__(self, path, types=None, size=None, mode="train", num_context=30, num_test=1, num_trials=9,
                 action_range=10, process_data=0):
        self.path = path
        self.types = types
        self.size = size
        self.mode = mode
        self.num_trials = num_trials
        self.num_context = num_context
        self.num_test = num_test
        self.action_range = action_range
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]
        types_str = '_'.join(self.types)

        self.path_list = []
        self.json_list = []
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
                    fx, fy = state[e][-1]['agent'][0]
                    if len(state[e][-1]['objects']) == 1:
                        gx, gy = state[e][-1]['objects'][0]
                    else:
                        min_dist = 10000
                        gx, gy = -1, -1
                        for o in state[e][-1]['objects']:
                            ox, oy = o[0]
                            d = ((ox - fx) ** 2 + (oy - fy) ** 2) ** 0.5
                            if d < min_dist:
                                gx, gy = ox, oy
                    for f in range(30, l - 83):
                        # find action taken; this calculation is approximate
                        f0x, f0y = state[e][f]['agent'][0]
                        f1x, f1y = state[e][f + 1]['agent'][0]
                        dx = (f1x - f0x) / 2.
                        dy = (f1y - f0y) / 2.
                        action = [dx, dy]
                        self.data_tuples[-1].append((v, past_len + f, action, [gx, gy], [f1x, f1y]))
                    print(len(self.data_tuples[-1]))
                    assert len(self.data_tuples[-1]) > 0
                    past_len += l

            index_dict = {'data_tuples': self.data_tuples}
            with open(os.path.join(self.path, f'index_bib_{mode}_{types_str}.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:
            with open(os.path.join(self.path, f'index_bib_{mode}_{types_str}.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_tuples = index_dict['data_tuples']

        self.tot_trials = len(self.path_list) * 9

    def _get_frame(self, video, frame_idx):
        cap = cv2.VideoCapture(video)
        # read frame at id and resize
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()
        if self.size is not None:
            assert frame is not None, f'frame is empty {frame_idx}, {video}'
            frame = cv2.resize(frame, self.size)
        frame = torch.tensor(frame).permute(2, 0, 1)
        # return frames as a torch tensor f x c x w x h
        frame = frame.to(torch.float32) / 255.
        cap.release()
        return frame

    def get_trial(self, trials, num_transitions, step=1):
        # retrieve state embeddings and actions from cached file
        states = []
        next_states = []
        actions = []
        rewards = []
        trial_len = []

        for t in trials:
            trial_len += [(t, n) for n in range(0, len(self.data_tuples[t]), step)]
        random.shuffle(trial_len)
        if len(trial_len) < num_transitions:
            return None, None, None, None, False

        for t, n in trial_len[:num_transitions]:
            video = self.data_tuples[t][n][0]
            states.append(self._get_frame(video, self.data_tuples[t][n][1]))
            next_states.append(self._get_frame(video, self.data_tuples[t][n + self.action_range][1]))

            goal_location = self.data_tuples[t][0][3]
            if len(self.data_tuples[t]) > n + self.action_range:
                actions_xy = [d[2] for d in self.data_tuples[t][n:n + self.action_range]]
                final_location = self.data_tuples[t][n + self.action_range][4]
            else:
                actions_xy = [d[2] for d in self.data_tuples[t][n:]]
                final_location = self.data_tuples[t][-1][4]
            distance_goal = ((goal_location[0] - final_location[0]) ** 2 + (
                    goal_location[1] - final_location[1]) ** 2) ** 0.5
            if distance_goal < 20:
                reward = - distance_goal
            else:
                reward = -100.
            actions_xy = np.array(actions_xy)
            actions_xy = np.mean(actions_xy, axis=0)
            actions.append(actions_xy)
            rewards.append(reward)
        states = torch.stack(states, dim=0)
        next_states = torch.stack(next_states, dim=0)
        rewards = torch.tensor(rewards)
        actions = torch.tensor(np.array(actions))
        return states, actions, next_states, rewards, True

    def __getitem__(self, idx):
        # retrieve 2 expert trajectories
        dem = False
        test = False
        while not dem or not test:
            ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]
            random.shuffle(ep_trials)
            dem_states, dem_actions, dem_next_states, dem_rewards, dem = self.get_trial(ep_trials[:-1],
                                                                                        self.num_context)
            test_states, test_actions, test_next_states, test_rewards, test = self.get_trial([ep_trials[-1]],
                                                                                             self.num_test,
                                                                                             step=self.action_range)
        return dem_states, dem_actions, dem_next_states, dem_rewards, test_states, test_actions, test_next_states, test_rewards

    def __len__(self):
        return self.tot_trials // self.num_trials

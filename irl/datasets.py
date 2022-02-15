import json
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data
import tqdm


def collate_function_seq(batch):
    dem_frames = torch.stack([item[0] for item in batch])
    dem_actions = torch.stack([item[1] for item in batch])
    dem_lens = [item[2] for item in batch]
    query_frames = torch.stack([item[3] for item in batch])
    target_actions = torch.stack([item[4] for item in batch])
    return [dem_frames, dem_actions, dem_lens, query_frames, target_actions]

def index_data(json_list, path_list):
    print(f'processing files {len(json_list)}')
    data_tuples = []
    for j, v in tqdm(zip(json_list, path_list)):
        with open(j, 'r') as f:
            state = json.load(f)
        ep_lens = [len(x) for x in state]
        past_len = 0
        for e, l in enumerate(ep_lens):
            data_tuples.append([])
            # skip first 30 frames and last 83 frames
            for f in range(30, l - 83):
                # find action taken; 
                f0x, f0y = state[e][f]['agent'][0]
                f1x, f1y = state[e][f + 1]['agent'][0]
                dx = (f1x - f0x) / 2.
                dy = (f1y - f0y) / 2.
                action = [dx, dy]
                # action = ACTION_LIST.index([dx, dy])
                data_tuples[-1].append((v, past_len + f, action))
            assert len(data_tuples[-1]) > 0
            past_len += l
    return data_tuples

    

class TransitionDataset(torch.utils.data.Dataset):
    """
    Training dataset class for the behavior cloning mlp model.
    Args:
        path: path to the dataset
        types: list of video types to include
        size: size of the frames to be returned
        mode: train, val
        num_context: number of context state-action pairs
        num_test: number of test state-action pairs
        num_trials: number of trials in an episode
        action_range: number of frames to skip; actions are combined over these number of frames (displcement) of the agent
        process_data: whether to the videos or not (skip if already processed)
    __getitem__:
        returns:  (dem_frames, dem_actions, query_frames, target_actions)
        dem_frames: (num_context, 3, size, size)
        dem_actions: (num_context, 2)
        query_frames: (num_test, 3, size, size)
        target_actions: (num_test, 2)
    """
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
        # get video paths and json file paths
        for t in types:
            print(f'reading files of type {t} in {mode}')
            paths = [os.path.join(self.path, t, x) for x in os.listdir(self.path) if
                     x.endswith(f'.mp4')]
            jsons = [os.path.join(self.path, t, x) for x in os.listdir(self.path) if
                     x.endswith(f'.json') and 'index' not in x]

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
            # index the videos in the dataset directory. This is done to speed up the retrieval of videos.
            # frame index, action tuples are stored
            self.data_tuples = index_data(self.json_list, self.path_list) 
            # tuples of frame index and action (displacement of agent) 
            index_dict = {'data_tuples': self.data_tuples}
            with open(os.path.join(self.path, f'index_bib_{mode}_{types_str}.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:
            # read pre-indexed data
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

            # actions are pooled over frames 
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
        # retrieve expert trajectories
        dem = False
        test = False
        # dem and test check for valid trajectory samples
        while not dem or not test:
            ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]
            random.shuffle(ep_trials)
            dem_states, dem_actions, dem = self.get_trial(ep_trials[:-1], self.num_context)
            test_states, test_actions, test = self.get_trial([ep_trials[-1]], self.num_test,
                                                             step=self.action_range)
        return dem_states, dem_actions, test_states, test_actions

    def __len__(self):
        return self.tot_trials // self.num_trials


class TestTransitionDataset(torch.utils.data.Dataset):
    """
    Test dataset class for the behavior cloning mlp model. This dataset is used to test the model on the eval data.
    This class is used to compare plausible and implausible episodes.
    Args:
        path: path to the dataset
        types: video type to evaluate on 
        size: size of the frames to be returned
        mode: test
        num_context: number of context state-action pairs
        num_test: number of test state-action pairs
        num_trials: number of trials in an episode
        action_range: number of frames to skip; actions are combined over these number of frames (displcement) of the agent
        process_data: whether to the videos or not (skip if already processed)
    __getitem__:
        returns:  (expected_dem_frames, expected_dem_actions, expected_query_frames, expected_target_actions,
        unexpected_dem_frames, unexpected_dem_actions, unexpected_query_frames, unexpected_target_actions)
        dem_frames: (num_context, 3, size, size)
        dem_actions: (num_context, 2)
        query_frames: (num_test, 3, size, size)
        target_actions: (num_test, 2)
    """
    def __init__(self, path, task_type=None, size=None, mode="test", num_context=30, num_test=1, num_trials=9,
                 action_range=10, process_data=0):
        self.path = path
        self.task_type = task_type
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

        print(f'reading files of type {task_type} in {mode}')
        paths_expected = sorted([os.path.join(self.path, task_type, x) for x in os.listdir(self.path) if
                                 x.endswith(f'e.mp4')])
        jsons_expected = sorted([os.path.join(self.path, task_type, x) for x in os.listdir(self.path) if
                                 x.endswith(f'e.json') and 'index' not in x])
        paths_unexpected = sorted([os.path.join(self.path, task_type, x) for x in os.listdir(self.path) if
                                   x.endswith(f'u.mp4')])
        jsons_unexpected = sorted([os.path.join(self.path, task_type, x) for x in os.listdir(self.path) if
                                   x.endswith(f'u.json') and 'index' not in x])

        self.path_list_exp += paths_expected
        self.json_list_exp += jsons_expected
        self.path_list_un += paths_unexpected
        self.json_list_un += jsons_unexpected

        self.data_unexpected = []
        self.data_expected = []
        if process_data:
            # index data. This is done speed up video retrieval.
            # frame index, action tuples are stored
            self.data_expected = self.index_data(self.json_list_exp, self.path_list_exp)
            index_dict = {'data_tuples': self.data_expected}
            with open(os.path.join(self.path, f'index_bib_test_{task_type}e.json'), 'w') as fp:
                json.dump(index_dict, fp)

            self.data_unexpected = self.index_data(self.json_list_un, self.path_list_un)
            index_dict = {'data_tuples': self.data_unexpected}
            with open(os.path.join(self.path, f'index_bib_test_{task_type}u.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:
            # load pre-indexed data
            with open(os.path.join(self.path, f'index_bib_test_{task_type}e.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_expected = index_dict['data_tuples']
            with open(os.path.join(self.path, f'index_bib_test_{task_type}u.json'), 'r') as fp:
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
        # retrieve expert trajectories
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


class TransitionDatasetSequence(torch.utils.data.Dataset):
    """
    Training dataset class for the behavior cloning mlp model.
    Args:
        path: path to the dataset
        types: list of video types to include
        size: size of the frames to be returned
        mode: train, val
        num_context: number of context state-action pairs
        num_test: number of test state-action pairs
        num_trials: number of trials in an episode
        action_range: number of frames to skip; actions are combined over these number of frames (displcement) of the agent
        process_data: whether to the videos or not (skip if already processed)
        max_len: maximum length of the sequence
    __getitem__:
        returns:  (dem_frames, dem_actions, dem_lens, query_frames, target_actions)
        dem_frames: (num_context, 3, size, size)
        dem_actions: (num_context, max_len, 2)
        dem_lens: (num_context)
        query_frames: (num_test, 3, size, size)
        target_actions: (num_test, 2)
    """

    def __init__(self, path, types=None, size=None, mode="train", num_context=30, num_test=1, num_trials=9,
                 action_range=10, process_data=0, max_len=30):
        self.path = path
        self.types = types
        self.size = size
        self.mode = mode
        self.num_trials = num_trials
        self.num_context = num_context
        self.num_test = num_test
        self.action_range = action_range
        self.max_len = max_len
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]
        types_str = '_'.join(self.types)

        self.path_list = []
        self.json_list = []
        for t in types:
            print(f'reading files of type {t} in {mode}')
            paths = [os.path.join(self.path, t, x) for x in os.listdir(self.path) if
                     x.endswith(f'.mp4')]
            jsons = [os.path.join(self.path, t, x) for x in os.listdir(self.path) if
                     x.endswith(f'.json') and 'index' not in x]

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
            # index the videos for quicker video retrieval.
            self.data_tuples = index_data(self.json_list, self.path_list)
            index_dict = {'data_tuples': self.data_tuples}
            with open(os.path.join(self.path, f'index_bib_{mode}_{types_str}.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:
            # load pre-existing index
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

    def get_trial(self, trials, step=10):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []
        lens = []

        for t in trials:
            tl = [(t, n) for n in range(0, len(self.data_tuples[t]), step)]
            if len(tl) > self.max_len:
                tl = tl[:self.max_len]
            trial_len.append(tl)

        for tl in trial_len:
            states.append([])
            actions.append([])
            lens.append(len(tl))
            for t, n in tl:
                video = self.data_tuples[t][n][0]
                states[-1].append(self._get_frame(video, self.data_tuples[t][n][1]))

                if len(self.data_tuples[t]) > n + self.action_range:
                    actions_xy = [d[2] for d in self.data_tuples[t][n:n + self.action_range]]
                else:
                    actions_xy = [d[2] for d in self.data_tuples[t][n:]]
                actions_xy = np.array(actions_xy)
                actions_xy = np.mean(actions_xy, axis=0)
                actions[-1].append(actions_xy)
            states[-1] = torch.stack(states[-1])
            trial_frames_padded = torch.zeros(self.max_len, states[-1].size(1), states[-1].size(2),
                                              states[-1].size(3))
            trial_frames_padded[:states[-1].size(0), :, :, :] = states[-1]
            states[-1] = trial_frames_padded
            actions[-1] = torch.tensor(np.array(actions[-1]))
            trial_actions_padded = torch.zeros(self.max_len, actions[-1].size(1))
            trial_actions_padded[:actions[-1].size(0), :] = actions[-1]
            actions[-1] = trial_actions_padded

        states = torch.stack(states)
        actions = torch.stack(actions)
        return states, actions, lens

    def get_test(self, trial, step=10):
        # retrieve state embeddings and actions from cached file
        trial_len = []
        trial_len += [(trial, n) for n in range(0, len(self.data_tuples[trial]), step)]

        query_idx = random.randint(0, len(trial_len) - 1)
        context_len = query_idx - 1
        tq, nq = trial_len[query_idx]

        video = self.data_tuples[tq][nq][0]
        query_frame = self._get_frame(video, self.data_tuples[tq][nq][1])
        if len(self.data_tuples[tq]) > nq + self.action_range:
            actions_xy = [d[2] for d in self.data_tuples[tq][nq:nq + self.action_range]]
        else:
            actions_xy = [d[2] for d in self.data_tuples[tq][nq:]]
        actions_xy = np.array(actions_xy)
        actions_xy = np.mean(actions_xy, axis=0)
        target_action = torch.tensor(actions_xy)
        # return trial_frames_padded, trial_actions_padded, context_len, query_frame, target_action
        return query_frame, target_action

    def __getitem__(self, idx):
        # retrieve expert trajectories
        ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]
        random.shuffle(ep_trials)
        dem_states, dem_actions, dem_lens = self.get_trial(ep_trials[:-1], step=self.action_range)
        query_frame, target_action = self.get_test(ep_trials[-1], step=self.action_range)
        return dem_states, dem_actions, dem_lens, query_frame, target_action

    def __len__(self):
        return self.tot_trials // self.num_trials


class TestTransitionDatasetSequence(torch.utils.data.Dataset):
    """
    Test dataset class for the behavior cloning rnn model. This dataset is used to test the model on the eval data.
    This class is used to compare plausible and implausible episodes.
    Args:
        path: path to the dataset
        types: list of video types to include
        size: size of the frames to be returned
        mode: test
        num_context: number of context state-action pairs
        num_test: number of test state-action pairs
        num_trials: number of trials in an episode
        action_range: number of frames to skip; actions are combined over these number of frames (displcement) of the agent
        process_data: whether to the videos or not (skip if already processed)
    __getitem__:
        returns:  (expected_dem_frames, expected_dem_actions, expected_dem_lens expected_query_frames, expected_target_actions,
        unexpected_dem_frames, unexpected_dem_actions, unexpected_dem_lens, unexpected_query_frames, unexpected_target_actions)
        dem_frames: (num_context, max_len, 3, size, size)
        dem_actions: (num_context, max_len, 2)
        dem_lens: (num_context)
        query_frames: (num_test, 3, size, size)
        target_actions: (num_test, 2)
    """
    def __init__(self, path, task_type=None, size=None, mode="train", num_context=30, num_test=1, num_trials=9,
                 action_range=10, process_data=0, max_len=30):
        self.path = path
        self.task_type = task_type
        self.size = size
        self.mode = mode
        self.num_trials = num_trials
        self.num_context = num_context
        self.num_test = num_test
        self.action_range = action_range
        self.max_len = max_len
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]


        self.path_list_exp = []
        self.json_list_exp = []
        self.path_list_un = []
        self.json_list_un = []

        print(f'reading files of type {task_type} in {mode}')
        paths_expected = sorted([os.path.join(self.path, task_type, x) for x in os.listdir(self.path) if
                                 x.endswith(f'{task_type}e.mp4')])
        jsons_expected = sorted([os.path.join(self.path, task_type, x) for x in os.listdir(self.path) if
                                 x.endswith(f'{task_type}e.json') and 'index' not in x])
        paths_unexpected = sorted([os.path.join(self.path, task_type, x) for x in os.listdir(self.path) if
                                   x.endswith(f'{task_type}u.mp4')])
        jsons_unexpected = sorted([os.path.join(self.path, task_type, x) for x in os.listdir(self.path) if
                                   x.endswith(f'{task_type}u.json') and 'index' not in x])

        self.path_list_exp += paths_expected
        self.json_list_exp += jsons_expected
        self.path_list_un += paths_unexpected
        self.json_list_un += jsons_unexpected


        self.data_expected = []
        self.data_unexpected = []

        if process_data:
            # index data. This is done to speed up video retrieval.
            # frame index, action tuples are stored
            self.data_expected = self.index_data(self.json_list_exp, self.path_list_exp)
            index_dict = {'data_tuples': self.data_expected}
            with open(os.path.join(self.path, f'index_bib_test_{task_type}e.json'), 'w') as fp:
                json.dump(index_dict, fp)

            self.data_unexpected = self.index_data(self.json_list_un, self.path_list_un)
            index_dict = {'data_tuples': self.data_unexpected}
            with open(os.path.join(self.path, f'index_bib_test_{task_type}u.json'), 'w') as fp:
                json.dump(index_dict, fp)

        else:
            with open(os.path.join(self.path, f'index_bib_{mode}_{task_type}e.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_expected = index_dict['data_tuples']
            with open(os.path.join(self.path, f'index_bib_{mode}_{task_type}u.json'), 'r') as fp:
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

    def get_trial(self, trials, data, step=10):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []
        lens = []

        for t in trials:
            tl = [(t, n) for n in range(0, len(data[t]), step)]
            if len(tl) > self.max_len:
                tl = tl[:self.max_len]
            trial_len.append(tl)

        for tl in trial_len:
            states.append([])
            actions.append([])
            lens.append(len(tl))
            for t, n in tl:
                video = data[t][n][0]
                states[-1].append(self._get_frame(video, data[t][n][1]))

                if len(data[t]) > n + self.action_range:
                    actions_xy = [d[2] for d in data[t][n:n + self.action_range]]
                else:
                    actions_xy = [d[2] for d in data[t][n:]]
                actions_xy = np.array(actions_xy)
                actions_xy = np.mean(actions_xy, axis=0)
                actions[-1].append(actions_xy)
            states[-1] = torch.stack(states[-1])
            trial_frames_padded = torch.zeros(self.max_len, states[-1].size(1), states[-1].size(2),
                                              states[-1].size(3))
            trial_frames_padded[:states[-1].size(0), :, :, :] = states[-1]
            states[-1] = trial_frames_padded
            actions[-1] = torch.tensor(np.array(actions[-1]))
            trial_actions_padded = torch.zeros(self.max_len, actions[-1].size(1))
            trial_actions_padded[:actions[-1].size(0), :] = actions[-1]
            actions[-1] = trial_actions_padded

        states = torch.stack(states)
        actions = torch.stack(actions)
        return states, actions, lens

    def get_test(self, trial, data, step=10):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []

        trial_len += [(trial, n) for n in range(0, len(data[trial]), step)]

        for t, n in trial_len:
            video = data[t][n][0]
            state = self._get_frame(video, data[t][n][1])

            if len(data[t]) > n + self.action_range:
                actions_xy = [d[2] for d in data[t][n:n + self.action_range]]
            else:
                actions_xy = [d[2] for d in data[t][n:]]
            actions_xy = np.array(actions_xy)
            actions_xy = np.mean(actions_xy, axis=0)
            actions.append(actions_xy)
            states.append(state)

        states = torch.stack(states)
        actions = torch.tensor(np.array(actions))
        return states, actions

    def __getitem__(self, idx):
        # retrieve expert trajectories
        ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]
        dem_expected_states, dem_expected_actions, dem_expected_lens = self.get_trial(ep_trials[:-1], self.data_expected, step=self.action_range)
        dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens = self.get_trial(ep_trials[:-1], self.data_unexpected, step=self.action_range)

        query_expected_frames, target_expected_actions = self.get_test(ep_trials[-1], self.data_expected, step=self.action_range)
        query_unexpected_frames, target_unexpected_actions = self.get_test(ep_trials[-1], self.data_unexpected, step=self.action_range)

        return dem_expected_states, dem_expected_actions, dem_expected_lens, query_expected_frames, target_expected_actions, \
                dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, query_unexpected_frames, target_unexpected_actions

    def __len__(self):
        return len(self.path_list_exp)


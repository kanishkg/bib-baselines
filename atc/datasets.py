import json
import os

import cv2
import torch
import torch.utils.data


class FrameDataset(torch.utils.data.Dataset):

    def __init__(self, path, types=None, size=None, shift=1, mode="train", process_data=1):
        self.path = path
        self.types = types
        self.size = size
        self.shift = shift
        self.mode = mode
        self.json_list = []
        self.path_list = []

        # read video files
        for t in types:
            print(f'reading files of type {t} in {mode}')
            self.path_list += [os.path.join(self.path, t, x) for x in os.listdir(os.path.join(self.path, t)) if
                               x.endswith(f'e.mp4')]
            self.json_list += [os.path.join(self.path, t, x) for x in os.listdir(os.path.join(self.path, t)) if
                               x.endswith(f'e.json')]

        self.path_list = sorted(self.path_list)
        self.json_list = sorted(self.json_list)

        # split for train and val
        if mode == 'train':
            self.path_list = self.path_list[:int(0.8 * len(self.path_list))]
            self.json_list = self.json_list[:int(0.8 * len(self.json_list))]
        elif mode == 'val':
            self.path_list = self.path_list[int(0.8 * len(self.path_list)):]
            self.json_list = self.json_list[int(0.8 * len(self.json_list)):]

        self.data_tuples = []

        # process json files to extract frame indices for training atc
        if process_data:
            # index videos to make frame retrieval easier
            print('processing files')
            for j, v in zip(self.json_list, self.path_list):
                print(j)
                try:
                    with open(j, 'r') as f:
                        state = json.load(f)
                except UnicodeDecodeError as e:
                    print(f'file skipped {j} with {e}')
                    continue
                ep_lens = [len(x) for x in state]
                past_len = 0
                for e, l in enumerate(ep_lens):
                    # skip first 30 frames and last 83 frames
                    for f in range(30, l - 83 - self.shift):
                        self.data_tuples.append((v, f + past_len, f + past_len + self.shift))
                    past_len += l
            index_dict = {'data_tuples': self.data_tuples}
            with open(os.path.join(self.path, f'index_dict_{mode}.json'), 'w') as fp:
                json.dump(index_dict, fp)
        else:
            with open(os.path.join(self.path, f'index_dict_{mode}.json'), 'r') as fp:
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
        video = self.data_tuples[idx][0]
        frames_idx = [self.data_tuples[idx][1], self.data_tuples[idx][2]]
        frames = self._get_frames(video, frames_idx)
        in_frame = frames[0, :, :, :]
        tar_frame = frames[1, :, :, :]
        return in_frame, tar_frame

    def __len__(self):
        return len(self.data_tuples)

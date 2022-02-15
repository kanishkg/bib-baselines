from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from irl.datasets import *
from irl.models import *

parser = ArgumentParser()

parser.add_argument('--model_type', type=str, default='bcmlp')
parser.add_argument('--ckpt', type=str, default=None, help='path to checkpoint')
parser.add_argument('--data_path', type=str, default=None, help='path to the data')
parser.add_argument('--size', type=int, default=84)


parser.add_argument('--surprise_type', type=str, default='mean',
                    help='surprise type: mean, max. This is used for comparing the plausibility scores of the two test episodes')
parser.add_argument('--types', nargs='+', type=str, default=['cl', 'ba'],
                    help='types of tasks used for training / testing')

args = parser.parse_args()

if args.model_type == 'bcmlp':
    model = BCMLP.load_from_checkpoint(args.ckpt)
elif args.model_type == 'bcrnn':
    model = BCRNN.load_from_checkpoint(args.ckpt)
model.eval()

for t in args.types:
    if args.model_type == 'bcmlp':
        test_dataset = TestTransitionDataset(args.dataset, task_type=t, size=(args.size, args.size), process_data=0, mode='test')
    elif args.model_type == 'bcrnn':
        test_dataset = TestTransitionDatasetSequence(args.dataset, task_type=t, size=(args.size, args.size), process_data=0, mode='test')

    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)
    count = 0
    pbar = tqdm(test_dataloader)
    for j, batch in enumerate(pbar):
        if args.model_type == 'bcmlp':
            dem_expected_states, dem_expected_actions, query_expected_frames, target_expected_actions, \
                dem_unexpected_states, dem_unexpected_actions, query_unexpected_frames, target_unexpected_actions = batch    
        elif args.model_type == 'bcrnn':
            dem_expected_states, dem_expected_actions, dem_expected_lens, query_expected_frames, target_expected_actions, \
                dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, query_unexpected_frames, target_unexpected_actions = batch

        # calculate the plausibility scores for the expected episode
        surprise_expected = []
        for i in range(query_expected_frames.size(1)):
            if args.model_type == 'bcmlp':
                test_actions, test_actions_pred = model(
                    [dem_expected_states, dem_expected_actions, query_expected_frames[:, i:i+1, :, :, :], target_expected_actions[:,i,:]])
            elif args.model_type == 'bcrnn':
                test_actions, test_actions_pred = model(
                    [dem_expected_states, dem_expected_actions, dem_expected_lens, query_expected_frames[:, i, :, :, :], target_expected_actions[:,i,:]])

            loss = F.mse_loss(test_actions, test_actions_pred)
            surprise_expected.append(loss.cpu().detach().numpy())

            mean_expected_surprise = np.mean(surprise_expected)
            max_expected_surprise = np.max(surprise_expected)

        # calculate the plausibility scores for the unexpected episode
        surprise_unexpected = []
        for i in range(query_unexpected_frames.size(1)):
            if args.model_type == 'bcmlp':
                test_actions, test_actions_pred = model(
                    [dem_unexpected_states, dem_unexpected_actions, query_unexpected_frames[:, i:i+1, :, :, :], target_unexpected_actions[:,i,:]]) 
            elif args.model_type == 'bcrnn':
                test_actions, test_actions_pred = model(
                    [dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, query_unexpected_frames[:, i, :, :, :], target_unexpected_actions[:,i,:]])
            loss = F.mse_loss(test_actions, test_actions_pred)
            surprise_unexpected.append(loss.cpu().detach().numpy())


        mean_unexpected_surprise = np.mean(surprise_unexpected)
        max_unexpected_surprise = np.max(surprise_unexpected)

        correct_mean = mean_expected_surprise < mean_unexpected_surprise + 0.5 * (
                    mean_expected_surprise == mean_unexpected_surprise)
        correct_max = max_expected_surprise < max_unexpected_surprise + 0.5 * (
                            max_expected_surprise == max_unexpected_surprise)
        if args.surprise_type == 'max':
            count += correct_max
        elif args.surprise_type == 'mean':
            count += correct_mean
        pbar.set_postfix({'accuracy': count/(j+1.), 'type': t})

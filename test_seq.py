import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

from irl.datasets import TestSeqTransitionDataset
from irl.models import ContextImitationLSTM
from irl.datasets import TestRawTransitionDataset
from irl.models import ContextImitationPixel

mode = 'rnn'
if mode=='mlp':
    model = ContextImitationPixel.load_from_checkpoint('lightning_logs/version_932742/checkpoints/epoch=26-step=9449.ckpt')
    test_datasets = TestRawTransitionDataset('/misc/vlgscratch4/LakeGroup/kanishk/bib-dataset/bib_eval_2/', types='ee', size=(84,84), process_data=1, mode='test')
elif mode=='rnn':
    model = ContextImitationLSTM.load_from_checkpoint('lightning_logs/version_947097/checkpoints/epoch=17-step=12599.ckpt')
    test_datasets = TestSeqTransitionDataset('/misc/vlgscratch4/LakeGroup/kanishk/bib-dataset/bib_eval_2/', types='ee', size=(84,84), process_data=0, mode='test')
model.eval()


test_dataloader = DataLoader(test_datasets, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)


c = 0
pbar = tqdm(test_dataloader)

for j, batch in enumerate(pbar):
    if mode == 'mlp':
        dem_expected_states, dem_expected_actions, query_expected_frames, target_expected_actions, \
            dem_unexpected_states, dem_unexpected_actions, query_unexpected_frames, target_unexpected_actions = batch

    elif mode == 'rnn':
        dem_expected_states, dem_expected_actions, dem_expected_lens, query_expected_frames, target_expected_actions, \
            dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, query_unexpected_frames, target_unexpected_actions = batch

    surprise_expected = []
    em = -1
    max_surp = 0
    te, teh = None, None

    for i in range(query_expected_frames.size(1)):
        if mode == 'mlp':
            test_actions, test_actions_pred = model(
                    [dem_expected_states, dem_expected_actions, query_expected_frames[:, i:i+1, :, :, :], target_expected_actions[:,i,:]])


        elif mode == 'rnn':
            test_actions, test_actions_pred = model(
                [dem_expected_states, dem_expected_actions, dem_expected_lens, query_expected_frames[:, i, :, :, :], target_expected_actions[:,i,:]])

        loss = F.mse_loss(test_actions, test_actions_pred)
        if loss.cpu().detach().numpy() > max_surp:
            max_surp = loss.cpu().detach().numpy()
            em = i
            te, teh = test_actions, test_actions_pred

        surprise_expected.append(loss.cpu().detach().numpy())

        mean_expected_surprise = np.mean(surprise_expected)
        max_expected_surprise = np.max(surprise_expected)


    surprise_unexpected = []
    um = -1
    max_surpu = 0
    tu, tuh = None, None

    for i in range(query_unexpected_frames.size(1)):
        if mode == 'mlp':
            test_actions, test_actions_pred = model(
                    [dem_unexpected_states, dem_unexpected_actions, query_unexpected_frames[:, i:i+1, :, :, :], target_unexpected_actions[:,i,:]])

        elif mode == 'rnn':
            test_actions, test_actions_pred = model(
                [dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, query_unexpected_frames[:, i, :, :, :], target_unexpected_actions[:,i,:]])


        loss = F.mse_loss(test_actions, test_actions_pred)
        if loss.cpu().detach().numpy() > max_surpu:
            max_surpu = loss.cpu().detach().numpy()
            um = i
            tu, tuh = test_actions, test_actions_pred

        surprise_unexpected.append(loss.cpu().detach().numpy())


    mean_unexpected_surprise = np.mean(surprise_unexpected)
    max_unexpected_surprise = np.max(surprise_unexpected)

    correct_mean = mean_expected_surprise < mean_unexpected_surprise + 0.5 * (
                mean_expected_surprise == mean_unexpected_surprise)
    correct_max = max_expected_surprise < max_unexpected_surprise + 0.5 * (
                        max_expected_surprise == max_unexpected_surprise)
    c+=correct_max
    pbar.set_postfix({'accuracy': c/(j+1.)})

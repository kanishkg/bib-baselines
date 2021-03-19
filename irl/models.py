import copy
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from atc.models import MlpModel


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared

class PEMIRL(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--latent_size', type=int, default=128)
        parser.add_argument('--anchor_size', type=int, default=256)
        parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 32, 32])
        parser.add_argument('--filter', nargs='+', type=int, default=[3, 3, 3, 3])
        parser.add_argument('--strides', nargs='+', type=int, default=[2, 2, 2, 1])
        parser.add_argument('--target_update_interval', type=int, default=1)
        parser.add_argument('--target_update_tau', type=float, default=0.01)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.lr = hparams.lr
        self.state_dim = hparams.state_dim
        self.action_dim = hparams.action_dim
        self.context_dim = hparams.action_dim

        self.context_enc_mean = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
                                         output_size=self.context_dim)
        self.context_enc_std = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
                                        output_size=self.context_dim)

        self.policy = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[64, 64],
                               output_size=self.action_dim)

        self.r = MlpModel(input_size=self.state_dim + self.context_dim + self.action_dim, hidden_sizes=[64, 32],
                          output_size=1)

        self.past_samples = []

    def training_step(self, batch, batch_idx):
        dem_states, dem_actions, test_states, test_actions, dem_l, test_l, test_mask = batch

        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.context_enc_mean(dem_traj)
        context_std_samples = self.context_enc_std(dem_traj)

        # combine contexts of each meta episode
        context_std_squared = torch.clamp(context_std_samples*context_std_samples, min=1e-7)
        context_std_squared = 1. / torch.sum(torch.reciprocal(context_std_squared), dim=1)
        context_mean = context_std_squared * torch.sum(context_mean_samples / context_std_squared, dim=1)
        context_std = torch.sqrt(context_std_squared)

        # sample context variable
        context_dist = torch.distributions.normal.Normal(context_mean, context_std)
        context = torch.normal(context_mean, context_std)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context, test_states], dim=1)
        test_context_states_actions = torch.cat([context, test_states, test_actions], dim=1)

        # for each state in the test states calculate action
        test_actions_pred = F.softmax(self.policy(test_context_states), dim=1)
        test_context_states_actions_pred = torch.cat([context, test_states, test_actions_pred], dim=1)

        # calculate the score of the expert trajectory using current policy
        test_actions_expert_score = test_actions * test_actions_pred

        # calculate the reward for each context, state, action that is observed
        test_reward_expert = self.r(test_context_states_actions)
        test_reward_pred = self.r(test_context_states_actions_pred)

        # calculate adversarial loss
        discriminator_expert_numerator = torch.exp(test_reward_expert)
        discriminator_expert_denominator = torch.exp(test_reward_expert) + test_actions_expert_score
        discriminator_pred_numerator = torch.exp(test_reward_pred)
        discriminator_pred_denominator = torch.exp(test_reward_pred) + test_actions_pred
        discriminator_loss = torch.log(discriminator_pred_numerator) - torch.log(
            discriminator_pred_denominator) + torch.log(discriminator_expert_denominator) - torch.log(
            discriminator_expert_numerator)

        # calculate info loss; used to calculate gradients for context encoder
        context_log_prob = context_dist.log_prob(context)
        info_loss = - context_log_prob

        # calculate info loss reward; used to calculate gradients for reward
        info_loss_reward = context_log_prob * test_reward_pred - context_log_prob * torch.mean(test_reward_pred, dim=1)

        # calculate policy likelihood loss for imitation
        imitation_loss = - torch.log(test_context_states_actions_pred * test_actions)

        # log metrics and optimize steps

        # add policy optimization

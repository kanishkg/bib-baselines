import math
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from atc.models import MlpModel
from irl.datasets import TransitionDataset, TestTransitionDataset


class TransformerModel(torch.nn.Module):

    def __init__(self, ntoken, nout, ninp=128, nhead=4, nhid=64, nlayers=2, dropout=0.):
        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.decoder = torch.nn.Linear(ninp, nout)

    def forward(self, src):
        src = F.relu(self.encoder(src)) * math.sqrt(self.ninp)
        output = self.transformer_encoder(src)
        output = F.relu(self.decoder(output))
        return output


class ContextImitation(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--state_dim', type=int, default=128)
        parser.add_argument('--action_dim', type=int, default=2)
        parser.add_argument('--beta', type=float, default=0.01)
        parser.add_argument('--gamma', type=float, default=0.0)
        parser.add_argument('--context_dim', nargs='+', type=int, default=32)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.lr = self.hparams.lr
        self.state_dim = self.hparams.state_dim
        self.action_dim = self.hparams.action_dim
        self.context_dim = self.hparams.context_dim
        self.beta = self.hparams.beta
        self.gamma = self.hparams.gamma

        # self.context_enc_mean = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
        #                                  output_size=self.context_dim)
        self.context_enc_mean = TransformerModel(self.state_dim + self.action_dim, nout=self.context_dim)

        # self.context_enc_std = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
        #                                 output_size=self.context_dim)

        # for param in self.context_enc_std.parameters():
        #     param.requires_grad = False
        # for param in self.context_enc_mean.parameters():
        #     param.requires_grad = False

        self.policy = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                               output_size=self.action_dim)

        self.past_samples = []

    def forward(self, batch):
        dem_states, dem_actions, test_states, test_actions = batch
        dem_states = dem_states.float()
        dem_actions = dem_actions.float()
        test_actions = test_actions.float()
        test_states = test_states.float()

        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.context_enc_mean(dem_traj)
        # context_std_samples = self.context_enc_std(dem_traj)

        # combine contexts of each meta episode

        # context_std_squared = torch.clamp(context_std_samples * context_std_samples, min=1e-7)
        # context_std_squared_reduced = 1. / torch.sum(torch.reciprocal(context_std_squared), dim=1)
        # context_mean = context_std_squared_reduced * torch.sum(context_mean_samples / context_std_squared, dim=1)
        # context_std = torch.sqrt(context_std_squared_reduced)

        # sample context variable
        # context_dist = torch.distributions.normal.Normal(context_mean, context_std)
        # prior_dist = torch.distributions.Normal(torch.zeros_like(context_mean), torch.ones_like(context_std))

        # context = torch.normal(context_mean, context_std)
        context = torch.mean(context_mean_samples, dim=1)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context.unsqueeze(1), test_states], dim=2)
        b, s, d = test_context_states.size()
        test_context_states = test_context_states.view(b * s, d)
        test_actions = test_actions.view(b * s, -1)

        # for each state in the test states calculate action
        test_actions_pred = F.tanh(self.policy(test_context_states))

        # calculate the test context distribution for the state and bring it closer to inferred context
        # test_states_actions = torch.cat([test_states, test_actions.view(b, s, -1)], dim=2)
        # test_context_mean_samples = self.context_enc_mean(test_states_actions)
        # test_context_std_samples = self.context_enc_std(test_states_actions)

        # combine contexts of test samples
        # test_context_std_squared = torch.clamp(test_context_std_samples * test_context_std_samples, min=1e-7)
        # test_context_std_squared_reduced = 1. / torch.sum(torch.reciprocal(test_context_std_squared), dim=1)
        # test_context_mean = test_context_std_squared_reduced * torch.sum(
        #     test_context_mean_samples / test_context_std_squared, dim=1)
        # test_context_std = torch.sqrt(test_context_std_squared_reduced)
        #
        # test_context_dist = torch.distributions.normal.Normal(test_context_mean, test_context_std)
        # test_context = torch.normal(test_context_mean, test_context_std)
        # test_context = torch.mean()

        return test_actions, test_actions_pred

    def training_step(self, batch, batch_idx):
        test_actions, test_actions_pred = self.forward(batch)
        loss = F.mse_loss(test_actions, test_actions_pred)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
        # calculate policy likelihood loss for imitation
        # imitation_loss = torch.mean(torch.sum(- torch.log(test_actions_pred + 1e-8) * test_actions, dim=1), dim=0)
        # kl_loss = torch.mean(torch.sum(context_dist.log_prob(context) - prior_dist.log_prob(context), dim=1), dim=0)
        # context_loss = torch.mean(torch.sum(- context_dist.log_prob(test_context), dim=1), dim=0)
        # loss = imitation_loss + self.beta * kl_loss + self.gamma * context_loss
        #
        # correct = torch.argmax(test_actions_pred.detach(), dim=1) == torch.argmax(test_actions.detach(),
        #                                                                           dim=1)
        # accuracy = torch.mean(correct.float())

        # self.log('imitation_loss', imitation_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('context_loss', context_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # if optimizer_idx == 0:
        # elif optimizer_idx == 1:
        #     return context_loss + self.beta * kl_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        test_actions, test_actions_pred = self.forward(batch)
        loss = F.mse_loss(test_actions, test_actions_pred)
        # calculate policy likelihood loss for imitation
        # imitation_loss = torch.mean(torch.sum(- torch.log(test_actions_pred + 1e-8) * test_actions, dim=1), dim=0)
        # kl_loss = torch.mean(torch.sum(context_dist.log_prob(context) - prior_dist.log_prob(context), dim=1), dim=0)
        # context_loss = torch.mean(
        #     torch.sum(test_context_dist.log_prob(test_context) - context_dist.log_prob(test_context), dim=1), dim=0)
        #
        # loss = imitation_loss + self.beta * kl_loss + self.gamma * context_loss
        #
        # correct = torch.argmax(test_actions_pred.detach(), dim=1) == torch.argmax(test_actions.detach(),
        #                                                                           dim=1)
        # accuracy = torch.mean(correct.float())

        self.log('val_loss', loss, on_epoch=True, logger=True)
        # self.log('val_imitation_loss', imitation_loss, prog_bar=True, logger=True)
        # self.log('val_kl_loss', kl_loss, prog_bar=True, logger=True)
        # self.log('val_accuracy', accuracy, prog_bar=True, logger=True)
        # self.log('val_context_loss', context_loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        fam_expected_states, fam_expected_actions, test_expected_states, test_expected_actions, \
        fam_unexpected_states, fam_unexpected_actions, test_unexpected_states, test_unexpected_actions = batch

        surprise_expected = []
        for i in range(test_expected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_expected_states, fam_expected_actions, test_expected_states[:, i, :].unsqueeze(1),
                 test_expected_actions[:, i, :].unsqueeze(1)])
            loss = F.mse_loss(test_actions, test_actions_pred)
            surprise_expected.append(loss.cpu().numpy())

        mean_expected_surprise = np.mean(surprise_expected)
        max_expected_surprise = np.max(surprise_expected)

        surprise_unexpected = []
        for i in range(test_unexpected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_unexpected_states, fam_unexpected_actions, test_unexpected_states[:, i, :].unsqueeze(1),
                 test_unexpected_actions[:, i, :].unsqueeze(1)])

            loss = F.mse_loss(test_actions, test_actions_pred)
            surprise_unexpected.append(loss.cpu().numpy())

        mean_unexpected_surprise = np.mean(surprise_unexpected)
        max_unexpected_surprise = np.max(surprise_unexpected)

        correct_mean = mean_expected_surprise < mean_unexpected_surprise + 0.5 * (
                mean_expected_surprise == mean_unexpected_surprise)
        correct_max = max_expected_surprise < max_unexpected_surprise + 0.5 * (
                max_expected_surprise == max_unexpected_surprise)
        self.log('test_expected_surprise', mean_expected_surprise, on_epoch=True, logger=True)
        self.log('test_unexpected_surprise', mean_unexpected_surprise, prog_bar=True, logger=True)
        self.log('test_expected_surprisem', max_expected_surprise, on_epoch=True, logger=True)
        self.log('test_unexpected_surprisem', max_unexpected_surprise, prog_bar=True, logger=True)
        self.log('accuracy_mean', correct_mean, prog_bar=True, logger=True)
        self.log('accuracy_max', correct_max, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        # policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        # context_optim = torch.optim.Adam(
        #     list(self.context_enc_mean.parameters()) + list(self.context_enc_std.parameters()), lr=self.lr)
        # return [policy_optim, context_optim]
        return optim

    def train_dataloader(self):
        train_dataset = TransitionDataset(self.hparams.data_path, types=self.hparams.types, mode='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_datasets = []
        val_loaders = []
        for t in self.hparams.types:
            val_datasets.append(TransitionDataset(self.hparams.data_path, types=[t], mode='val'))
            val_loaders.append(DataLoader(dataset=val_datasets[-1], batch_size=self.hparams.batch_size,
                                          num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False))
        return val_loaders

    def test_dataloader(self):
        test_datasets = []
        test_dataloaders = []
        for t in self.hparams.types:
            test_datasets.append(TestTransitionDataset(self.hparams.data_path, type=t))
            test_dataloaders.append(
                DataLoader(dataset=test_datasets[-1], batch_size=1, num_workers=1, pin_memory=True, shuffle=False))
        return test_dataloaders


class IRLNoDynamics(pl.LightningModule):

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

        self.lr = self.hparams.lr
        self.state_dim = self.hparams.state_dim
        self.action_dim = self.hparams.action_dim
        self.context_dim = self.hparams.action_dim
        self.beta = self.hparams.beta

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
        context_std_squared = torch.clamp(context_std_samples * context_std_samples, min=1e-7)
        context_std_squared = 1. / torch.sum(torch.reciprocal(context_std_squared), dim=1)
        context_mean = context_std_squared * torch.sum(context_mean_samples / context_std_squared, dim=1)
        context_std = torch.sqrt(context_std_squared)

        # sample context variable
        context_dist = torch.distributions.normal.Normal(context_mean, context_std)
        prior_dist = torch.distributions.Normal(torch.zeros_like(context_mean), torch.ones_like(context_std))

        context = torch.normal(context_mean, context_std)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context, test_states], dim=1)
        test_context_states_actions = torch.cat([context, test_states, test_actions], dim=1)

        # for each state in the test states calculate action
        test_actions_pred = F.softmax(self.policy(test_context_states), dim=1)
        test_context_states_actions_pred = torch.cat([context, test_states, test_actions_pred], dim=1)

        # calculate policy likelihood loss for imitation
        imitation_loss = - torch.log(test_context_states_actions_pred + 1e-8) * test_actions
        kl_loss = context_dist.log_prob(context) - prior_dist.log_prob(context)
        loss = imitation_loss + self.beta * kl_loss

        # calculate the score of the expert trajectory using current policy
        # test_actions_expert_score = test_actions * test_actions_pred

        # calculate the reward for each context, state, action that is observed
        # test_reward_expert = self.r(test_context_states_actions)
        # test_reward_pred = self.r(test_context_states_actions_pred)

        # calculate adversarial loss
        # discriminator_expert_numerator = torch.exp(test_reward_expert)
        # discriminator_expert_denominator = torch.exp(test_reward_expert) + test_actions_expert_score
        # discriminator_pred_numerator = torch.exp(test_reward_pred)
        # discriminator_pred_denominator = torch.exp(test_reward_pred) + test_actions_pred
        # discriminator_loss = torch.log(discriminator_pred_numerator) - torch.log(
        #     discriminator_pred_denominator) + torch.log(discriminator_expert_denominator) - torch.log(
        #     discriminator_expert_numerator)

        # calculate info loss; used to calculate gradients for context encoder
        # context_log_prob = context_dist.log_prob(context)
        # info_loss = - context_log_prob

        # calculate info loss reward; used to calculate gradients for reward
        # info_loss_reward = context_log_prob * test_reward_pred - context_log_prob * torch.mean(test_reward_pred, dim=1)

        # log metrics and optimize steps

        # add policy optimization

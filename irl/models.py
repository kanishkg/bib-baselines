import copy
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from atc.models import MlpModel
from irl.datasets import TransitionDataset


class ContextImitation(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--state_dim', type=int, default=128)
        parser.add_argument('--action_dim', type=int, default=9)
        parser.add_argument('--beta', type=float, default=0.01)
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

        self.context_enc_mean = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
                                         output_size=self.context_dim)
        self.context_enc_std = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
                                        output_size=self.context_dim)

        self.policy = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[64, 64],
                               output_size=self.action_dim)

        self.past_samples = []

    def training_step(self, batch, batch_idx):
        dem_states, dem_actions, test_states, test_actions = batch
        dem_states = dem_states.float()
        dem_actions = dem_actions.float()
        test_actions = test_actions.float()
        test_states = test_states.float()

        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.context_enc_mean(dem_traj)
        context_std_samples = self.context_enc_std(dem_traj)

        # combine contexts of each meta episode

        context_std_squared = torch.clamp(context_std_samples * context_std_samples, min=1e-7)
        context_std_squared_reduced = 1. / torch.sum(torch.reciprocal(context_std_squared), dim=1)
        context_mean = context_std_squared_reduced * torch.sum(context_mean_samples / context_std_squared, dim=1)
        context_std = torch.sqrt(context_std_squared_reduced)

        # sample context variable
        context_dist = torch.distributions.normal.Normal(context_mean, context_std)
        prior_dist = torch.distributions.Normal(torch.zeros_like(context_mean), torch.ones_like(context_std))

        context = torch.normal(context_mean, context_std)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context.unsqueeze(1), test_states], dim=2)
        b, s, d = test_context_states.size()
        test_context_states = test_context_states.view(b * s, d)
        test_actions = test_actions.view(b * s, -1)

        # for each state in the test states calculate action
        test_actions_pred = F.softmax(self.policy(test_context_states), dim=1)

        # calculate policy likelihood loss for imitation
        imitation_loss = torch.mean(torch.sum(- torch.log(test_actions_pred + 1e-8) * test_actions, dim=1), dim=0)
        kl_loss = torch.mean(torch.sum(context_dist.log_prob(context) - prior_dist.log_prob(context), dim=1), dim=0)

        loss = imitation_loss + self.beta * kl_loss

        correct = torch.argmax(test_actions_pred.detach(), dim=1) == torch.argmax(test_actions.detach(),
                                                                                  dim=1)
        accuracy = torch.mean(correct.float())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('imitation_loss', imitation_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        dem_states, dem_actions, test_states, test_actions = batch

        dem_states = dem_states.float()
        dem_actions = dem_actions.float()
        test_actions = test_actions.float()
        test_states = test_states.float()

        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)
        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.context_enc_mean(dem_traj)
        context_std_samples = self.context_enc_std(dem_traj)

        # combine contexts of each meta episode

        context_std_squared = torch.clamp(context_std_samples * context_std_samples, min=1e-7)
        context_std_squared_reduced = 1. / torch.sum(torch.reciprocal(context_std_squared), dim=1)
        context_mean = context_std_squared_reduced * torch.sum(context_mean_samples / context_std_squared, dim=1)
        context_std = torch.sqrt(context_std_squared_reduced)

        # sample context variable batch x context-size
        context_dist = torch.distributions.normal.Normal(context_mean, context_std)
        prior_dist = torch.distributions.Normal(torch.zeros_like(context_mean), torch.ones_like(context_std))

        context = torch.normal(context_mean, context_std)

        # concat context embedding to the state embedding of test trajectory batch x test-samples x dim+context-size
        test_context_states = torch.cat([context.unsqueeze(1), test_states], dim=2)
        b, s, d = test_context_states.size()
        test_context_states = test_context_states.view(b * s, d)
        test_actions = test_actions.view(b * s, -1)

        # for each state in the test states calculate action
        test_actions_pred = F.softmax(self.policy(test_context_states), dim=1)

        # calculate policy likelihood loss for imitation
        imitation_loss = torch.mean(torch.sum(- torch.log(test_actions_pred + 1e-8) * test_actions, dim=1), dim=0)
        kl_loss = torch.mean(torch.sum(context_dist.log_prob(context) - prior_dist.log_prob(context), dim=1), dim=0)
        loss = imitation_loss + self.beta * kl_loss

        correct = torch.argmax(test_actions_pred.detach(), dim=1) == torch.argmax(test_actions.detach(),
                                                                                  dim=1)
        accuracy = torch.mean(correct.float())

        self.log(f'val_loss', loss, on_epoch=True, logger=True)
        self.log(f'val_imitation_loss', imitation_loss, prog_bar=True, logger=True)
        self.log(f'val_kl_loss', kl_loss, prog_bar=True, logger=True)
        self.log(f'val_accuracy', accuracy, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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

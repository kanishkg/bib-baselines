import math
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from atc.models import MlpModel, EncoderModel, ATCEncoder
from atc.utils import update_state_dict
from irl.datasets import TransitionDataset, TestTransitionDataset, RawTransitionDataset, TestRawTransitionDataset, \
    RewardTransitionDataset


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
        parser.add_argument('--dropout', type=float, default=0.2)

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
        self.dropout = self.hparams.dropout

        # self.context_enc_mean = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
        #                                  output_size=self.context_dim)
        self.context_enc_mean = TransformerModel(self.state_dim + self.action_dim, nout=self.context_dim,
                                                 dropout=self.dropout)

        # self.context_enc_std = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
        #                                 output_size=self.context_dim)

        # for param in self.context_enc_std.parameters():
        #     param.requires_grad = False
        # for param in self.context_enc_mean.parameters():
        #     param.requires_grad = False

        self.policy = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                               output_size=self.action_dim, dropout=self.dropout)

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


class ContextImitationPixel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--state_dim', type=int, default=128)
        parser.add_argument('--action_dim', type=int, default=2)
        parser.add_argument('--beta', type=float, default=0.01)
        parser.add_argument('--gamma', type=float, default=0.0)
        parser.add_argument('--context_dim', nargs='+', type=int, default=32)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--size', type=int, default=84)
        parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 32, 32])
        parser.add_argument('--filter', nargs='+', type=int, default=[3, 3, 3, 3])
        parser.add_argument('--strides', nargs='+', type=int, default=[2, 2, 2, 1])
        parser.add_argument('--process_data', type=int, default=0)

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
        self.dropout = self.hparams.dropout

        # self.encoder = EncoderModel(image_shape=[3, self.hparams.size, self.hparams.size],
        #                             latent_size=self.hparams.state_dim,
        #                             channels=self.hparams.channels, kernel_sizes=self.hparams.filter,
        #                             strides=self.hparams.strides)
        self.encoder = ATCEncoder.load_from_checkpoint(
            '/data/kvg245/bib-tom/lightning_logs/version_911764/checkpoints/epoch=31-step=342118.ckpt')
        self.context_enc_mean = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
                                         output_size=self.context_dim)
        # self.context_enc_mean = TransformerModel(self.state_dim + self.action_dim, nout=self.context_dim,
        #                                          dropout=self.dropout)

        # for param in self.context_enc_std.parameters():
        #     param.requires_grad = False
        # for param in self.context_enc_mean.parameters():
        #     param.requires_grad = False

        self.policy = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                               output_size=self.action_dim, dropout=self.dropout)

        self.past_samples = []

    def forward(self, batch):
        dem_frames, dem_actions, test_frames, test_actions = batch
        dem_frames = dem_frames.float()
        dem_actions = dem_actions.float()
        test_actions = test_actions.float()
        test_frames = test_frames.float()

        dem_states, _ = self.encoder.encoder(dem_frames)
        test_states, _ = self.encoder.encoder(test_frames)
        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.context_enc_mean(dem_traj)
        context = torch.mean(context_mean_samples, dim=1)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context.unsqueeze(1), test_states], dim=2)
        b, s, d = test_context_states.size()
        test_context_states = test_context_states.view(b * s, d)
        test_actions = test_actions.view(b * s, -1)

        # for each state in the test states calculate action
        test_actions_pred = F.tanh(self.policy(test_context_states))

        return test_actions, test_actions_pred

    def training_step(self, batch, batch_idx):
        test_actions, test_actions_pred = self.forward(batch)
        loss = F.mse_loss(test_actions, test_actions_pred)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        test_actions, test_actions_pred = self.forward(batch)
        loss = F.mse_loss(test_actions, test_actions_pred)
        self.log('val_loss', loss, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        fam_expected_states, fam_expected_actions, test_expected_states, test_expected_actions, \
        fam_unexpected_states, fam_unexpected_actions, test_unexpected_states, test_unexpected_actions = batch

        surprise_expected = []
        for i in range(test_expected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_expected_states, fam_expected_actions, test_expected_states[:, i, :, :, :].unsqueeze(1),
                 test_expected_actions[:, i, :].unsqueeze(1)])
            loss = F.mse_loss(test_actions, test_actions_pred)
            surprise_expected.append(loss.cpu().numpy())

        mean_expected_surprise = np.mean(surprise_expected)
        max_expected_surprise = np.max(surprise_expected)

        surprise_unexpected = []
        for i in range(test_unexpected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_unexpected_states, fam_unexpected_actions, test_unexpected_states[:, i, :, :, :].unsqueeze(1),
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
        return optim

    def train_dataloader(self):
        train_dataset = RawTransitionDataset(self.hparams.data_path, types=self.hparams.types, mode='train',
                                             process_data=self.hparams.process_data,
                                             size=(self.hparams.size, self.hparams.size))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_datasets = []
        val_loaders = []
        for t in self.hparams.types:
            val_datasets.append(RawTransitionDataset(self.hparams.data_path, types=[t], mode='val',
                                                     process_data=self.hparams.process_data,
                                                     size=(self.hparams.size, self.hparams.size)))
            val_loaders.append(DataLoader(dataset=val_datasets[-1], batch_size=self.hparams.batch_size,
                                          num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False))
        return val_loaders

    def test_dataloader(self):
        test_datasets = []
        test_dataloaders = []
        for t in self.hparams.types:
            test_datasets.append(
                TestRawTransitionDataset(self.hparams.data_path, types=t, process_data=self.hparams.process_data,
                                         size=(self.hparams.size, self.hparams.size)))
            test_dataloaders.append(
                DataLoader(dataset=test_datasets[-1], batch_size=1, num_workers=1, pin_memory=True, shuffle=False))
        return test_dataloaders


class ContextNLL(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--state_dim', type=int, default=128)
        parser.add_argument('--action_dim', type=int, default=2)
        parser.add_argument('--beta', type=float, default=0.001)
        parser.add_argument('--gamma', type=float, default=0.0)
        parser.add_argument('--context_dim', nargs='+', type=int, default=32)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--size', type=int, default=84)
        parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 32, 32])
        parser.add_argument('--filter', nargs='+', type=int, default=[3, 3, 3, 3])
        parser.add_argument('--strides', nargs='+', type=int, default=[2, 2, 2, 1])
        parser.add_argument('--process_data', type=int, default=0)

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
        self.dropout = self.hparams.dropout

        self.encoder = ATCEncoder.load_from_checkpoint(
            '/data/kvg245/bib-tom/lightning_logs/version_911764/checkpoints/epoch=31-step=342118.ckpt')
        self.context_enc_mean = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
                                         output_size=self.context_dim)

        self.policy_mean = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                                    output_size=self.action_dim, dropout=self.dropout)
        self.policy_std = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                                   output_size=self.action_dim, dropout=self.dropout)

        # self.discriminator = torch.nn.Sequential(
        #     MlpModel(input_size=self.state_dim + self.context_dim + self.action_dim,
        #              hidden_sizes=[256, 128, 256],
        #              output_size=1, dropout=self.dropout), torch.nn.Sigmoid())

        self.past_samples = []

    def forward(self, batch):
        dem_frames, dem_actions, test_frames, test_actions = batch
        dem_frames = dem_frames.float()
        dem_actions = dem_actions.float()
        test_actions = test_actions.float()
        test_frames = test_frames.float()

        dem_states, _ = self.encoder.encoder(dem_frames)
        test_states, _ = self.encoder.encoder(test_frames)
        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.context_enc_mean(dem_traj)

        # combine contexts of each meta episode
        context = torch.mean(context_mean_samples, dim=1)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context.unsqueeze(1), test_states], dim=2)
        b, s, d = test_context_states.size()
        test_context_states = test_context_states.view(b * s, d)
        test_actions = test_actions.view(b * s, -1)

        # for each state in the test states calculate action
        test_actions_pred_mu = torch.tanh(self.policy_mean(test_context_states))
        test_actions_pred_sig = torch.sigmoid(self.policy_std(test_context_states))

        return test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context

    def training_step(self, batch, batch_idx):
        test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context = self.forward(
            batch)
        test_actions_pred = torch.normal(test_actions_pred_mu, test_actions_pred_sig)
        policy_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
        entropy = torch.mean(policy_dist.entropy())
        nll = torch.mean(-policy_dist.log_prob(test_actions))
        loss = nll - self.beta * entropy
        mse_loss = F.mse_loss(test_actions, test_actions_pred)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('nll', nll, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('entropy', entropy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context = self.forward(batch)
        test_actions_pred = torch.normal(test_actions_pred_mu, test_actions_pred_sig)
        policy_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
        entropy = torch.mean(policy_dist.entropy())
        nll = torch.mean(-policy_dist.log_prob(test_actions))
        loss = nll - self.beta * entropy
        mse_loss = F.mse_loss(test_actions, test_actions_pred)
        self.log('val_mse_loss', mse_loss, on_epoch=True, logger=True)
        self.log('val_gen', loss, on_epoch=True, logger=True)
        self.log('val_nll', nll, on_epoch=True, logger=True)
        self.log('val_entropy', entropy, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        fam_expected_states, fam_expected_actions, test_expected_states, test_expected_actions, \
        fam_unexpected_states, fam_unexpected_actions, test_unexpected_states, test_unexpected_actions = batch

        surprise_expected = []
        for i in range(test_expected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_expected_states, fam_expected_actions, test_expected_states[:, i, :, :, :].unsqueeze(1),
                 test_expected_actions[:, i, :].unsqueeze(1)])
            loss = F.mse_loss(test_actions, test_actions_pred)
            surprise_expected.append(loss.cpu().numpy())

        mean_expected_surprise = np.mean(surprise_expected)
        max_expected_surprise = np.max(surprise_expected)

        surprise_unexpected = []
        for i in range(test_unexpected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_unexpected_states, fam_unexpected_actions, test_unexpected_states[:, i, :, :, :].unsqueeze(1),
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
        return optim

    def train_dataloader(self):
        train_dataset = RawTransitionDataset(self.hparams.data_path, types=self.hparams.types, mode='train',
                                             process_data=self.hparams.process_data,
                                             size=(self.hparams.size, self.hparams.size))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_datasets = []
        val_loaders = []
        for t in self.hparams.types:
            val_datasets.append(RawTransitionDataset(self.hparams.data_path, types=[t], mode='val',
                                                     process_data=self.hparams.process_data,
                                                     size=(self.hparams.size, self.hparams.size)))
            val_loaders.append(DataLoader(dataset=val_datasets[-1], batch_size=self.hparams.batch_size,
                                          num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False))
        return val_loaders

    def test_dataloader(self):
        test_datasets = []
        test_dataloaders = []
        for t in self.hparams.types:
            test_datasets.append(
                TestRawTransitionDataset(self.hparams.data_path, types=t, process_data=self.hparams.process_data,
                                         size=(self.hparams.size, self.hparams.size)))
            test_dataloaders.append(
                DataLoader(dataset=test_datasets[-1], batch_size=1, num_workers=1, pin_memory=True, shuffle=False))
        return test_dataloaders


class ContextAIL(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--state_dim', type=int, default=128)
        parser.add_argument('--action_dim', type=int, default=2)
        parser.add_argument('--beta', type=float, default=0.001)
        parser.add_argument('--gamma', type=float, default=0.0)
        parser.add_argument('--context_dim', nargs='+', type=int, default=32)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--size', type=int, default=84)
        parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 32, 32])
        parser.add_argument('--filter', nargs='+', type=int, default=[3, 3, 3, 3])
        parser.add_argument('--strides', nargs='+', type=int, default=[2, 2, 2, 1])
        parser.add_argument('--process_data', type=int, default=0)

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
        self.dropout = self.hparams.dropout

        self.generator = ContextNLL.load_from_checkpoint(
            '/data/kvg245/bib-tom/lightning_logs/version_932019/checkpoints/epoch=19-step=4999.ckpt')

        self.discriminator = torch.nn.Sequential(
            MlpModel(input_size=self.state_dim + self.context_dim + self.action_dim,
                     hidden_sizes=[256, 128, 256],
                     output_size=1, dropout=self.dropout), torch.nn.Sigmoid())

        self.past_samples = []

    def forward(self, batch):
        dem_frames, dem_actions, test_frames, test_actions = batch
        dem_frames = dem_frames.float()
        dem_actions = dem_actions.float()
        test_actions = test_actions.float()
        test_frames = test_frames.float()

        dem_states, _ = self.generator.encoder.encoder(dem_frames)
        test_states, _ = self.generator.encoder.encoder(test_frames)
        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.generator.context_enc_mean(dem_traj)

        # combine contexts of each meta episode
        context = torch.mean(context_mean_samples, dim=1)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context.unsqueeze(1), test_states], dim=2)
        b, s, d = test_context_states.size()
        test_context_states = test_context_states.view(b * s, d)
        test_actions = test_actions.view(b * s, -1)

        # for each state in the test states calculate action
        test_actions_pred_mu = torch.tanh(self.policy_mean(test_context_states))
        test_actions_pred_sig = torch.sigmoid(self.policy_std(test_context_states))

        return test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context = self.generator(
                batch)
            test_actions_pred = torch.normal(test_actions_pred_mu, test_actions_pred_sig)
            test_context_states_actions = torch.cat([test_context_states, test_actions], dim=1)
            test_context_states_actions_pred = torch.cat([test_context_states, test_actions_pred], dim=1)
            neg_disc = self.discriminator(test_context_states_actions)
            pos_disc = self.discriminator(test_context_states_actions_pred)
            disc_loss = - torch.mean(torch.log(pos_disc + 1e-8)) - torch.mean(torch.log(1 - neg_disc + 1e-8))
            self.log('disc_loss', disc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return disc_loss

        elif optimizer_idx == 1:
            test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context = self.generator(
                batch)
            test_actions_pred = torch.normal(test_actions_pred_mu, test_actions_pred_sig)
            test_context_states_actions_pred = torch.cat([test_context_states, test_actions_pred], dim=1)
            pos_disc = self.discriminator(test_context_states_actions_pred)

            policy_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
            entropy = torch.mean(policy_dist.entropy())
            nll = torch.mean(-policy_dist.log_prob(test_actions))
            gen_loss = torch.mean(torch.log(pos_disc + 1e-8)) - self.beta * entropy
            mse_loss = F.mse_loss(test_actions, test_actions_pred)

            self.log('gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('nll', nll, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('entropy', entropy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return gen_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context = self.generator(
            batch)
        test_actions_pred = torch.normal(test_actions_pred_mu, test_actions_pred_sig)
        test_context_states_actions = torch.cat([test_context_states, test_actions], dim=1)
        test_context_states_actions_pred = torch.cat([test_context_states, test_actions_pred], dim=1)
        neg_disc = self.discriminator(test_context_states_actions)
        pos_disc = self.discriminator(test_context_states_actions_pred)
        disc_loss = torch.mean(torch.log(pos_disc + 1e-8)) + torch.mean(torch.log(1 - neg_disc + 1e-8))
        policy_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
        entropy = torch.mean(policy_dist.entropy())
        nll = torch.mean(-policy_dist.log_prob(test_actions))
        gen_loss = - torch.mean(pos_disc) - self.beta * entropy
        mse_loss = F.mse_loss(test_actions, test_actions_pred)

        self.log('val_mse_loss', mse_loss, on_epoch=True, logger=True)
        self.log('val_loss', gen_loss, on_epoch=True, logger=True)
        self.log('val_disc', disc_loss, on_epoch=True, logger=True)
        self.log('val_nll', nll, on_epoch=True, logger=True)
        self.log('val_entropy', entropy, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        fam_expected_states, fam_expected_actions, test_expected_states, test_expected_actions, \
        fam_unexpected_states, fam_unexpected_actions, test_unexpected_states, test_unexpected_actions = batch

        surprise_expected = []
        for i in range(test_expected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_expected_states, fam_expected_actions, test_expected_states[:, i, :, :, :].unsqueeze(1),
                 test_expected_actions[:, i, :].unsqueeze(1)])
            loss = F.mse_loss(test_actions, test_actions_pred)
            surprise_expected.append(loss.cpu().numpy())

        mean_expected_surprise = np.mean(surprise_expected)
        max_expected_surprise = np.max(surprise_expected)

        surprise_unexpected = []
        for i in range(test_unexpected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_unexpected_states, fam_unexpected_actions, test_unexpected_states[:, i, :, :, :].unsqueeze(1),
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
        disc_optim = torch.optim.Adam(
            list(self.discriminator.parameters()) + list(self.generator.encoder.parameters()) + list(
                self.generator.context_enc_mean.parameters()), lr=self.lr)
        gen_optim = torch.optim.Adam(list(self.generator.policy_std.parameters()) + list(
            self.generator.policy_mean.parameters()), lr=self.lr)
        return [disc_optim, gen_optim]

    def train_dataloader(self):
        train_dataset = RawTransitionDataset(self.hparams.data_path, types=self.hparams.types, mode='train',
                                             process_data=self.hparams.process_data,
                                             size=(self.hparams.size, self.hparams.size))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_datasets = []
        val_loaders = []
        for t in self.hparams.types:
            val_datasets.append(RawTransitionDataset(self.hparams.data_path, types=[t], mode='val',
                                                     process_data=self.hparams.process_data,
                                                     size=(self.hparams.size, self.hparams.size)))
            val_loaders.append(DataLoader(dataset=val_datasets[-1], batch_size=self.hparams.batch_size,
                                          num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False))
        return val_loaders

    def test_dataloader(self):
        test_datasets = []
        test_dataloaders = []
        for t in self.hparams.types:
            test_datasets.append(
                TestRawTransitionDataset(self.hparams.data_path, types=t, process_data=self.hparams.process_data,
                                         size=(self.hparams.size, self.hparams.size)))
            test_dataloaders.append(
                DataLoader(dataset=test_datasets[-1], batch_size=1, num_workers=1, pin_memory=True, shuffle=False))
        return test_dataloaders


class OfflineRL(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--state_dim', type=int, default=128)
        parser.add_argument('--action_dim', type=int, default=2)
        parser.add_argument('--beta', type=float, default=0.001)
        parser.add_argument('--gamma', type=float, default=0.995)
        parser.add_argument('--context_dim', nargs='+', type=int, default=32)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--size', type=int, default=84)
        parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 32, 32])
        parser.add_argument('--filter', nargs='+', type=int, default=[3, 3, 3, 3])
        parser.add_argument('--strides', nargs='+', type=int, default=[2, 2, 2, 1])
        parser.add_argument('--process_data', type=int, default=0)
        parser.add_argument('--eps', type=float, default=0.1)
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
        self.dropout = self.hparams.dropout
        self.eps = self.hparams.eps

        self.encoder = ATCEncoder.load_from_checkpoint(
            '/data/kvg245/bib-tom/lightning_logs/version_911764/checkpoints/epoch=31-step=342118.ckpt')
        self.context_enc_mean = MlpModel(self.state_dim * 2 + self.action_dim + 1, hidden_sizes=[64, 64],
                                         output_size=self.context_dim)

        self.policy_mean_prior = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                                          output_size=self.action_dim, dropout=self.dropout)
        self.policy_std_prior = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                                         output_size=self.action_dim, dropout=self.dropout)

        self.policy_mean = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                                    output_size=self.action_dim, dropout=self.dropout)
        self.policy_std = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                                   output_size=self.action_dim, dropout=self.dropout)

        self.qnet = MlpModel(input_size=self.state_dim + self.action_dim + self.context_dim,
                             hidden_sizes=[256, 128, 256], output_size=1)
        self.qnet_target = MlpModel(input_size=self.state_dim + self.action_dim + self.context_dim,
                                    hidden_sizes=[256, 128, 256], output_size=1)

        self.eta = torch.nn.Parameter(torch.tensor(3.))
        self.alpha = torch.nn.Parameter(torch.tensor(1.))
        self.policy_dist_old = None

    def forward(self, batch):
        dem_frames, dem_actions, dem_next_frames, dem_r, test_frames, test_actions, test_next_frames, test_r, done = batch

        dem_frames = dem_frames.float()
        dem_actions = dem_actions.float()
        dem_next_frames = dem_next_frames.float()
        dem_r = dem_r.float()
        test_actions = test_actions.float()
        test_frames = test_frames.float()
        test_next_frames = test_next_frames.float()
        test_r = test_r.float()

        dem_states, _ = self.encoder.encoder(dem_frames)
        test_states, _ = self.encoder.encoder(test_frames)
        dem_next_states, _ = self.encoder.encoder(dem_next_frames)
        test_next_states, _ = self.encoder.encoder(test_next_frames)

        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions, dem_next_states, dem_r.unsqueeze(2)], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.context_enc_mean(dem_traj)

        # combine contexts of each meta episode
        context = torch.mean(context_mean_samples, dim=1)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context.unsqueeze(1), test_states], dim=2)
        b, s, d = test_context_states.size()
        test_context_states = test_context_states.view(b * s, d)
        test_actions = test_actions.view(b * s, -1)
        test_r = test_r.view(b * s, -1)
        done = done.view(b * s, -1)

        # for each state in the test states calculate action
        test_actions_pred_mu = torch.tanh(self.policy_mean_prior(test_context_states))
        test_actions_pred_sig = torch.sigmoid(self.policy_std_prior(test_context_states))

        return test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context, test_r, done

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context, \
            test_r, done = self.forward(batch)
            prior_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
            prior_loss = torch.mean(-prior_dist.log_prob(test_actions))
            self.log('prior_loss', prior_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return prior_loss

        elif optimizer_idx == 1:
            test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context, \
            test_r, done = self.forward(batch)
            prior_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
            actions_20 = prior_dist.sample_n(20)
            actions_20 = actions_20.permute(1, 0, 2)
            test_context_states_20 = test_context_states.unsqueeze(1).repeat(1, 20, 1)
            test_context_states_actions_20 = torch.cat([test_context_states_20, actions_20], dim=2)
            target_value = torch.mean(self.qnet_target(test_context_states_actions_20), dim=2)
            target_q_value = test_r + (1 - done) * self.gamma * target_value
            test_context_states_actions = torch.cat([test_context_states, test_actions], dim=1)
            qvalue = self.qnet(test_context_states_actions)
            qloss = F.mse_loss(qvalue, torch.mean(target_q_value.detach(), dim=1).unsqueeze(1))
            self.log('q_loss', qloss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return qloss

        elif optimizer_idx == 2:
            test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context, \
            test_r, done = self.forward(batch)
            prior_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
            actions_20 = prior_dist.sample_n(20)
            actions_20 = actions_20.permute(1, 0, 2)
            test_context_states_20 = test_context_states.unsqueeze(1).repeat(1, 20, 1)
            test_context_states_actions_20 = torch.cat([test_context_states_20, actions_20], dim=2)
            target_value = self.qnet_target(test_context_states_actions_20)
            eta = torch.sigmoid(self.eta)*3
            loss = torch.sum(eta * (self.eps + torch.log(torch.mean(torch.exp(target_value / eta), dim=1))))
            self.log('eta_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        elif optimizer_idx == 3:
            test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context, \
            test_r, done = self.forward(batch)
            prior_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
            actions_20 = prior_dist.sample_n(20)
            actions_20 = actions_20.permute(1, 0, 2)

            test_actions_mu = torch.tanh(self.policy_mean(test_context_states))
            test_actions_sig = torch.sigmoid(self.policy_std(test_context_states))
            policy_dist = torch.distributions.normal.Normal(test_actions_mu, test_actions_sig)

            test_context_states_20 = test_context_states.unsqueeze(1).repeat(1, 20, 1)
            test_context_states_actions_20 = torch.cat([test_context_states_20, actions_20], dim=2)
            target_value = self.qnet_target(test_context_states_actions_20)
            eta = torch.sigmoid(self.eta)*3
            alpha = torch.sigmoid(self.alpha)

            if self.policy_dist_old == None:
                self.policy_dist_old = torch.distributions.normal.Normal(test_actions_mu, test_actions_sig)

            log_prob = []
            for i in range(20):
                log_prob.append(policy_dist.log_prob(actions_20[:, i, :]))
            log_prob = torch.stack(log_prob, dim=1)
            loss = -torch.mean(
                torch.sum(torch.exp(target_value / eta) * log_prob, dim=1) + alpha * (
                        self.eps - torch.distributions.kl.kl_divergence(policy_dist, self.policy_dist_old)))
            self.log('policy_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.policy_dist_old = torch.distributions.normal.Normal(test_actions_mu.detach(),
                                                                     test_actions_sig.detach())
            return loss

        elif optimizer_idx == 4:
            test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context, \
            test_r, done = self.forward(batch)
            prior_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
            actions_20 = prior_dist.sample_n(20)
            actions_20 = actions_20.permute(1, 0, 2)

            test_actions_mu = torch.tanh(self.policy_mean(test_context_states))
            test_actions_sig = torch.sigmoid(self.policy_std(test_context_states))
            policy_dist = torch.distributions.normal.Normal(test_actions_mu, test_actions_sig)

            test_context_states_20 = test_context_states.unsqueeze(1).repeat(1, 20, 1)
            test_context_states_actions_20 = torch.cat([test_context_states_20, actions_20], dim=2)
            target_value = self.qnet_target(test_context_states_actions_20)
            eta = torch.sigmoid(self.eta)*3
            alpha = torch.sigmoid(self.alpha)
            log_prob = []
            for i in range(20):
                log_prob.append(policy_dist.log_prob(actions_20[:, i, :]))
            log_prob = torch.stack(log_prob, dim=1)

            loss = torch.mean(
                torch.sum(torch.exp(target_value / eta) * log_prob, dim=1) + alpha * (
                        self.eps - torch.distributions.kl.kl_divergence(policy_dist, self.policy_dist_old)))
            self.log('alpha_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            if batch_idx % 100 == 0:
                update_state_dict(self.qnet_target, self.qnet_target, 1)

            return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        test_context_states, test_actions, test_actions_pred_mu, test_actions_pred_sig, context, \
        q1, q2, value, test_r, done = self.forward(batch)
        policy_dist = torch.distributions.normal.Normal(test_actions_pred_mu, test_actions_pred_sig)
        test_actions_pred = torch.normal(test_actions_pred_mu, test_actions_pred_sig)
        predicted_value = self.valuenet(test_context_states)

        target_value = self.valuenet_target(test_context_states)
        target_q_value = test_r + (1 - done) * self.gamma * target_value

        q1loss = F.mse_loss(q1, target_q_value.detach())
        q2loss = F.mse_loss(q2, target_q_value.detach())

        test_context_states_actions_pred = torch.cat([test_context_states, test_actions_pred], dim=1)
        predicted_q = torch.min(self.softqnet1(test_context_states_actions_pred),
                                self.softqnet2(test_context_states_actions_pred))
        target_value_func = predicted_q - torch.sum(policy_dist.log_prob(test_actions_pred))

        value_loss = F.mse_loss(predicted_value, target_value_func.detach())

        policy_loss = torch.mean(torch.sum(policy_dist.log_prob(test_actions_pred)) - predicted_q)

        self.log('val_q1loss', q1loss, on_epoch=True, logger=True)
        self.log('val_q2loss', q2loss, on_epoch=True, logger=True)
        self.log('val_value_loss', value_loss, on_epoch=True, logger=True)
        self.log('val_policy_loss', policy_loss, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        fam_expected_states, fam_expected_actions, test_expected_states, test_expected_actions, \
        fam_unexpected_states, fam_unexpected_actions, test_unexpected_states, test_unexpected_actions = batch

        surprise_expected = []
        for i in range(test_expected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_expected_states, fam_expected_actions, test_expected_states[:, i, :, :, :].unsqueeze(1),
                 test_expected_actions[:, i, :].unsqueeze(1)])
            loss = F.mse_loss(test_actions, test_actions_pred)
            surprise_expected.append(loss.cpu().numpy())

        mean_expected_surprise = np.mean(surprise_expected)
        max_expected_surprise = np.max(surprise_expected)

        surprise_unexpected = []
        for i in range(test_unexpected_states.size(1)):
            test_actions, test_actions_pred = self.forward(
                [fam_unexpected_states, fam_unexpected_actions, test_unexpected_states[:, i, :, :, :].unsqueeze(1),
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
        prior_opt = torch.optim.Adam(list(self.context_enc_mean.parameters()) + list(self.encoder.parameters()) + list(
            self.policy_std_prior.parameters()) + list(self.policy_mean_prior.parameters()),
                                     lr=self.lr)
        q_opt = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        policy_opt = torch.optim.Adam(list(self.policy_std.parameters()) + list(self.policy_mean.parameters()),
                                      lr=self.lr)
        eta_opt = torch.optim.Adam([self.eta], lr=self.lr)
        alpha_opt = torch.optim.Adam([self.alpha], lr=self.lr)

        return [prior_opt, q_opt, eta_opt, policy_opt, alpha_opt]

    def train_dataloader(self):
        train_dataset = RewardTransitionDataset(self.hparams.data_path, types=self.hparams.types, mode='train',
                                                process_data=self.hparams.process_data,
                                                size=(self.hparams.size, self.hparams.size))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_datasets = []
        val_loaders = []
        for t in self.hparams.types:
            val_datasets.append(RewardTransitionDataset(self.hparams.data_path, types=[t], mode='val',
                                                        process_data=self.hparams.process_data,
                                                        size=(self.hparams.size, self.hparams.size)))
            val_loaders.append(DataLoader(dataset=val_datasets[-1], batch_size=self.hparams.batch_size,
                                          num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False))
        return val_loaders

    def test_dataloader(self):
        test_datasets = []
        test_dataloaders = []
        for t in self.hparams.types:
            test_datasets.append(
                TestRawTransitionDataset(self.hparams.data_path, types=t, process_data=self.hparams.process_data,
                                         size=(self.hparams.size, self.hparams.size)))
            test_dataloaders.append(
                DataLoader(dataset=test_datasets[-1], batch_size=1, num_workers=1, pin_memory=True, shuffle=False))
        return test_dataloaders

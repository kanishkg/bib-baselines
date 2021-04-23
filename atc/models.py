import copy
from argparse import ArgumentParser

import torch
from torch import nn
import pytorch_lightning as pl

from atc.utils import conv2d_output_shape, infer_leading_dims, restore_leading_dims, update_state_dict, random_shift


class MlpModel(nn.Module):
    """Multilayer Perceptron with last layer linear.
    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            dropout=None # Dropout value
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [nn.Linear(n_in, n_out) for n_in, n_out in
                         zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for l, layer in enumerate(hidden_layers):
            if dropout is not None and l != (len(hidden_sizes)-1):
                sequence.extend([layer, nonlinearity(), nn.Dropout(dropout)])
            else:
                 sequence.extend([layer, nonlinearity()])

        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
                             else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size


class Conv2dModel(nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
    ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [nn.Conv2d(in_channels=ic, out_channels=oc,
                                 kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
                       zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                                           child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c


class EncoderModel(nn.Module):

    def __init__(
            self,
            image_shape,
            latent_size,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            hidden_sizes=None,  # usually None; NOT the same as anchor MLP
    ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=False,
        )
        self._output_size = self.conv.conv_out_size(h, w)
        self.head = MlpModel(
            input_size=self._output_size,
            hidden_sizes=hidden_sizes,
            output_size=latent_size,
        )

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.view(T * B, *img_shape))
        c = self.head(conv.view(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)

        return c, conv  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size


class ContrastModel(nn.Module):

    def __init__(self, latent_size, anchor_hidden_sizes):
        super().__init__()
        if anchor_hidden_sizes is not None:
            self.anchor_mlp = MlpModel(
                input_size=latent_size,
                hidden_sizes=anchor_hidden_sizes,
                output_size=latent_size,
            )
        else:
            self.anchor_mlp = None
        self.W = nn.Linear(latent_size, latent_size, bias=False)

    def forward(self, anchor, positive):
        lead_dim, T, B, _ = infer_leading_dims(anchor, 1)
        assert lead_dim == 1  # Assume [B,C] shape
        if self.anchor_mlp is not None:
            anchor = anchor + self.anchor_mlp(anchor)  # skip probably helps
        pred = self.W(anchor)
        logits = torch.matmul(pred, positive.T)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # normalize
        return logits


class ATCEncoder(pl.LightningModule):

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
        print(hparams)
        self.hparams = hparams
        self.lr = self.hparams.lr
        self.target_update_interval = self.hparams.target_update_interval
        self.target_update_tau = self.hparams.target_update_tau
        self.encoder = EncoderModel(image_shape=[3, self.hparams.size, self.hparams.size], latent_size=self.hparams.latent_size,
                                    channels=self.hparams.channels, kernel_sizes=self.hparams.filter, strides=self.hparams.strides)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.contrast_model = ContrastModel(self.hparams.latent_size, self.hparams.anchor_size)
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, input):
        embedding, _ = self.encoder(input)
        return embedding

    def training_step(self, batch, batch_idx):
        if batch_idx % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(),
                              self.target_update_tau)
        xa, xp = batch
        if self.hparams.random_shift:
            xa = random_shift(
                imgs=xa,
                pad=1,
                prob=1.,
            )
            xp = random_shift(
                imgs=xp,
                pad=1,
                prob=1.,
            )
        with torch.no_grad():
            z_positive, _ = self.target_encoder(xp)
        z_anchor, conv_output = self.encoder(xa)
        logits = self.contrast_model(anchor=z_anchor, positive=z_positive)
        labels = torch.arange(z_anchor.shape[0],
                              dtype=torch.long, device=self.device)

        loss = self.celoss(logits, labels)
        correct = torch.argmax(logits.detach(), dim=1) == labels
        accuracy = torch.mean(correct.float())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        xa, xp = batch
        z_positive, _ = self.target_encoder(xp)
        z_anchor, conv_output = self.encoder(xa)
        logits = self.contrast_model(anchor=z_anchor, positive=z_positive)
        labels = torch.arange(z_anchor.shape[0],
                              dtype=torch.long, device=self.device)

        loss = self.celoss(logits, labels)
        correct = torch.argmax(logits.detach(), dim=1) == labels
        accuracy = torch.mean(correct.float())

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

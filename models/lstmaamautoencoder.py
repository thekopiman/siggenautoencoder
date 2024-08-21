import torch
import torch.nn as nn
import torch.optim as optim
import math


class AngularMargin(nn.Module):
    """
    An implementation of Angular Margin (AM) proposed in the following
    paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
    Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similiarity
    scale : float
        The scale for cosine similiarity

    Return
    ---------
    predictions : torch.Tensor

    Example
    -------
    >>> pred = AngularMargin()
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0):
        super(AngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        """Compute AM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : torch.Tensor
        """
        outputs = outputs - self.margin * targets
        return self.scale * outputs

class AdditiveAngularMargin(AngularMargin):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similiarity.
    scale: float
        The scale for cosine similiarity.

    Returns
    -------
    predictions : torch.Tensor
        Tensor.
    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> pred = AdditiveAngularMargin()
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super(AdditiveAngularMargin, self).__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : torch.Tensor
        """
        cosine = outputs.float()
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs

class LSTMAAMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, pre_defined_classes = 4):
        super(LSTMAAMAutoencoder, self).__init__()
        
        self.pre_defined_classes = pre_defined_classes
        
        self.encoder = nn.LSTM(input_dim, latent_dim, num_layers, batch_first=True)
        self.BN_latent = nn.BatchNorm1d(latent_dim)
        self.latent_classes = nn.Parameter(
            torch.FloatTensor(pre_defined_classes, latent_dim)        
        )
        nn.init.xavier_uniform_(self.latent_classes)

        self.decoder = nn.LSTM(latent_dim, input_dim, num_layers, batch_first=True)
        self.AAM = AdditiveAngularMargin()
        
        

    def forward(self, x, pre_target = None):
        # Encode
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1]  # Get the hidden state from the last layer
        after_bn = self.BN_latent(latent)
        
        if(self.training):
            assert(pre_target != None and pre_target.max() < self.pre_defined_classes)
            
            lst = []
            for idx, i in enumerate(pre_target):
                # print(after_bn[idx,:].shape, self.latent_classes[int(i):].shape)
                lst.append(self.AAM(after_bn[idx,:], self.latent_classes[int(i),:]))

            after_bn = torch.stack(lst)

        # Prepare repeated latent vector for decoder input
        latent_repeated = after_bn.unsqueeze(1).repeat(1, x.shape[-2], 1)

        # Decode
        output, _ = self.decoder(latent_repeated)

        return output, after_bn


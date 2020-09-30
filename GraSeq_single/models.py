import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
torch.backends.cudnn.enabled=False


def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class SelfAttention(nn.Module):
    """
    The class is an implementation of the multi-head self attention
    "A Structured Self-Attentive Sentence Embedding including regularization"
    https://arxiv.org/abs/1703.03130 in ICLR 2017
    We made light modifications for speedup
    """

    def __init__(self, hidden):
        super().__init__()

        self.first_linear = nn.Linear(hidden, 16)
        self.first_linear.bias.data.fill_(0)
        self.second_linear = nn.Linear(16, 1)
        self.second_linear.bias.data.fill_(0)

    def forward(self, encoder_outputs):

        # (B, Length, H) -> (B , Length, 10)
        first_hidden = self.first_linear(encoder_outputs)
        energy = self.second_linear(torch.tanh(first_hidden))

        attention = F.softmax(energy, dim=1).transpose(1, 2) # (B, 10, Length)
        # encoder_outputs is (B, Length, Hidden)
        sentence_embeddings = attention @ encoder_outputs
        outputs = sentence_embeddings.sum(dim=1)
        return outputs

    #Regularization
    def l2_matrix_norm(self, m):
        """
        Frobenius norm calculation
        Args:
           m: {Variable} ||AAT - I||
        Returns:
            regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)

'''
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs
'''

class encoder(nn.Module):

    def __init__(self, input_dim, hidden_size, latent_size, device):
        super(encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = nn.GRU(self.input_dim, int(self.hidden_size / 2), batch_first=True, bidirectional=True)

        self._mu = nn.Linear(in_features=self.hidden_size, out_features=latent_size)
        self._logvar = nn.Linear(in_features=self.hidden_size, out_features=latent_size)
        # self.attention = SelfAttention(self.hidden_size)

        self.apply(weights_init)
        self.to(device)

    def forward(self, x):

        x = x.reshape(1, -1, self.input_dim)
        outputs, last_hidden = self.encoder(x)

        # attn_output, attn_weights = self.attention(output)

        output = torch.mean(outputs, dim=1, keepdim=True)
        mu, logvar =  self._mu(output), self._logvar(output)

        encoder_outputs = outputs.squeeze(0)

        # print(encoder_outputs.shape, mu.shape, logvar.shape)

        return encoder_outputs, mu, logvar


class decoder(nn.Module):

    def __init__(self, latent_size, output_size, device):
        super(decoder, self).__init__()

        self.latent_size = latent_size
        self.output_size = output_size

        self.decoder = nn.GRU(self.latent_size, self.output_size, batch_first=True)
        self.to(device)

    def forward(self, x, sos, y):

        hidden = x.reshape(1, -1, self.latent_size)
        input = sos.reshape(1, -1, self.latent_size)

        expected_outputs = torch.zeros(y.shape)

        for di in range(0, y.shape[0]):
            output, hidden = self.decoder(input, hidden)

            # Teaching Forcing
            input = y[di, :].reshape(-1, 1, self.latent_size)
            expected_outputs[di, :] = output.reshape(-1, self.latent_size)

        return expected_outputs


class classifier(nn.Module):

    def __init__(self, latent_size, device):
        super(classifier,self).__init__()

        self.latent_size = latent_size

        self.classifier = nn.Sequential(nn.Linear(self.latent_size, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, 2),
                                        )

        self.apply(weights_init)
        self.to(device)

    def forward(self, x):

        out = self.classifier(x)
        return out

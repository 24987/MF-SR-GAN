import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
from SPM import SpectralNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F
from miscc.config import cfg


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        return cnn_code


# ############## G networks ###################

class GMapping(nn.Module):
    def __init__(self, dim_z, dim_w):
        super(GMapping, self).__init__()
        self.mapping = nn.Linear(dim_z, dim_w * 8 * 4 * 4)

    def forward(self, z):
        w = self.mapping(z)
        w_map = w.view(z.size(0), -1, 4, 4)
        return w_map


class ModulationBlock(nn.Module):
    def __init__(self, nf):
        super(ModulationBlock, self).__init__()
        self.fc_weight_1 = nn.Linear(256, 256)
        self.fc_weight_2 = nn.Linear(256, nf)
        self.fc_bias_1 = nn.Linear(256, 256)
        self.fc_bias_2 = nn.Linear(256, nf)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, code=None):

        weight = self.fc_weight_2(self.act(self.fc_weight_1(code)))
        bias = self.fc_bias_2(self.act(self.fc_bias_1(code)))

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class UpSamplingBlock_PixelShuffle(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super(UpSamplingBlock_PixelShuffle, self).__init__()
        self.flag = in_ch != out_ch
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.m1 = ModulationBlock(in_ch)
        self.m4 = ModulationBlock(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        if self.flag:
            self.add_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0))

        self.upModule = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, code):
        output = self.add(x) + self.gamma * self.modulate(x, code)

        if self.upsample:
            output = self.upModule(output)
        return output

    def add(self, x):
        if self.flag:
            x = self.add_conv(x)
        return x

    def modulate(self, x, code):
        y = self.act(self.m1(x, code))
        y = self.conv1(y)
        y = self.act(self.m4(y, code))
        y = self.conv2(y)
        return y


class UpSamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super(UpSamplingBlock, self).__init__()
        self.flag = in_ch != out_ch
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.m1 = ModulationBlock(in_ch)
        self.m4 = ModulationBlock(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        if self.flag:
            self.add_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0))

    def forward(self, x, code):
        output = self.add(x) + self.gamma * self.modulate(x, code)

        if self.upsample:
            output = F.interpolate(output, scale_factor=2)
        return output

    def add(self, x):
        if self.flag:
            x = self.add_conv(x)
        return x

    def modulate(self, x, code):
        y = self.act(self.m1(x, code))
        y = self.conv1(y)
        y = self.act(self.m4(y, code))
        y = self.conv2(y)
        return y


class NetG(nn.Module):
    def __init__(self, dim_w=32, dim_z=100):
        super(NetG, self).__init__()
        nf = dim_w
        self.gmap = GMapping(dim_z, dim_w)
        self.up1 = UpSamplingBlock(nf * 8, nf * 8)
        self.up2 = UpSamplingBlock(nf * 8, nf * 8)
        self.up3 = UpSamplingBlock(nf * 8, nf * 8)
        self.up4 = UpSamplingBlock(nf * 8, nf * 8)
        self.up5 = UpSamplingBlock(nf * 8, nf * 4)
        self.up6 = UpSamplingBlock(nf * 4, nf * 2)
        self.up7 = UpSamplingBlock(nf * 2, nf * 1, upsample=False)
        self.up8 = UpSamplingBlock(nf * 1, nf * 1, upsample=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(nf, 3, 3, 1, 1)
        self.act = nn.Tanh()

    def forward(self, x, c):
        out = self.gmap(x)
        out = self.up1(out, c)
        out = self.up2(out, c)
        out = self.up3(out, c)
        out = self.up4(out, c)
        out = self.up5(out, c)
        out = self.up6(out, c)
        out = self.up7(out, c)
        out = self.relu(self.up8(out, c))
        out = self.act(self.conv(out))
        return out


# ############## D networks ##########################
#
class DownSamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(DownSamplingBlock, self).__init__()
        self.flag = (in_ch != out_ch)
        self.downsample = downsample
        self.conv1 = SpectralNorm(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.add_conv = SpectralNorm(nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0))

    def forward(self, x):
        output = self.act(self.conv1(x))
        output = self.act(self.conv2(output))
        return self.add(x) + self.gamma * output

    def add(self, x):
        if self.flag:
            x = self.add_conv(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.embedding_dim = cfg.TEXT.EMBEDDING_DIM
        self.jointConv = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )
        self.embedding = nn.Sequential(SpectralNorm(nn.Conv2d(ndf * 16, ndf * 8, 3, 1, 1, bias = False)),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Conv2d(ndf * 8, self.embedding_dim, 4, 1, 0, bias=False))

    def forward(self, h_code, c_code=None):
        c_code = c_code.view(-1, 256, 1, 1).repeat(1, 1, 4, 4)
        h_c_code = torch.cat((h_code, c_code), 1)

        real_fake = self.jointConv(h_c_code)
        if h_code.size(0) == self.batch_size:
            sentence_embedding = self.embedding(h_code)
            sentence_embedding = sentence_embedding.view(self.batch_size, -1)
        else:
            sentence_embedding = None
        return sentence_embedding, real_fake

class NetD(nn.Module):
    def __init__(self, nf):
        super(NetD, self).__init__()
        self.conv = SpectralNorm(nn.Conv2d(3, nf, 3, 1, 1))  # 128
        self.down1 = DownSamplingBlock(nf * 1, nf * 2)  # 64
        self.down2 = DownSamplingBlock(nf * 2, nf * 4)  # 32
        self.down3 = DownSamplingBlock(nf * 4, nf * 8)  # 16
        self.down4 = DownSamplingBlock(nf * 8, nf * 16)  # 8
        self.down5 = DownSamplingBlock(nf * 16, nf * 16)  # 4
        self.down6 = DownSamplingBlock(nf * 16, nf * 16)  # 4
        self.COND_DNET = D_GET_LOGITS(nf)

    def forward(self, x):
        out = self.conv(x)
        out = self.down1(out)
        out = self.down2(out)
        out = self.down3(out)
        out = self.down4(out)
        out = self.down5(out)
        out = self.down6(out)
        return out

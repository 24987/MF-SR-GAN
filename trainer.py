from __future__ import print_function
from six.moves import range

import torch
from model import NetG, NetD
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init, load_params, copy_G_params

from model import RNN_ENCODER, CNN_ENCODER

import os
import time
from masks import mask_correlated_samples
import numpy as np
import torch.nn as nn


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks.bool(), -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.DAMSM_EPOCH = cfg.DAMSM_EPOCH

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        # image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        # img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        # state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        # image_encoder.load_state_dict(state_dict)
        # for p in image_encoder.parameters():
        #     p.requires_grad = False
        # print('Load image encoder from:', img_encoder_path)
        # image_encoder.eval()
        image_encoder = None

        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netG = NetG(cfg.TRAIN.NF, 100)
        netD = NetD(cfg.TRAIN.NF)
        netG.apply(weights_init)
        netD.apply(weights_init)

        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                s_tmp = Gname[:Gname.rfind('/')]
                Dname = '../models/netD_epoch_400.pth' # '%s/netD.pth' % (s_tmp)
                print('Load D from: ', Dname)
                state_dict = torch.load(Dname, map_location=lambda storage, loc: storage)
                netD.load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            # image_encoder = image_encoder.cuda()
            netG.cuda()
            netD.cuda()

        return [text_encoder, image_encoder, netG, netD, epoch]

    def define_optimizers(self, netG, netD):
        optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))
        return optimizerG, optimizerD

    def save_model(self, netG, avg_param_G, netD, epoch):
        torch.save(netG.state_dict(), '%s/current_netG_epoch_%d.pth' % (self.model_dir, epoch))
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(), '%s/average_netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)

        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.model_dir, epoch))
        print('Save G/Ds models.')

    def train_fake_real(self, prepare_data):
        text_encoder, image_encoder, netG, netD, start_epoch = self.build_models()
        netD.train()
        netG.train()
        #avg = NetG(cfg.TRAIN.NF, 100).cuda()
        #avg .load_state_dict(torch.load("../models/average_netG_epoch_400.pth"))
        avg_param_G = copy_G_params(netG)
        # del avg
        # torch.cuda.empty_cache()
        optimizerG, optimizerD = self.define_optimizers(netG, netD)

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = torch.FloatTensor(batch_size, nz)
        fixed_noise = torch.FloatTensor(batch_size, nz).normal_(0, 1)
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        match_labels = torch.LongTensor(range(batch_size)).cuda()

        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            average_gloss = []
            average_dloss = []
            average_sloss = []

            while step < self.num_batches:

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                _, sent_emb = text_encoder(captions, cap_lens, hidden)
                sent_emb = sent_emb.detach()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs = netG(noise, sent_emb)

                #######################################################
                # (3) Update D network
                ######################################################
                optimizerD.zero_grad()
                real_features = netD(imgs[0])

                real_embedding, output_right = netD.COND_DNET(real_features, sent_emb)
                errD_real = torch.nn.ReLU()(1.0 - output_right).mean()
                _, output_wrong = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
                errD_mismatch = torch.nn.ReLU()(1.0 + output_wrong).mean()

                fake_features = netD(fake_imgs.detach())
                _, errD_fake = netD.COND_DNET(fake_features, sent_emb)
                errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

                real_recon_loss0, real_recon_loss1 = sent_loss(real_embedding, sent_emb, match_labels, class_ids, batch_size)
                errD = errD_real + (errD_fake + errD_mismatch) / 2.0 + (real_recon_loss0 + real_recon_loss1) * 0.2
                errD.backward()
                optimizerD.step()
                D_logs = 'errD: %.2f Drecon: %.2f' % (errD.item(), (real_recon_loss0 + real_recon_loss1) * 0.2)

                step += 1
                gen_iterations += 1

                optimizerG.zero_grad()
                features = netD(fake_imgs)
                fake_embedding, output = netD.COND_DNET(features, sent_emb)
                #fake_recon_loss0, fake_recon_loss1 = sent_loss(fake_embedding, sent_emb, match_labels, class_ids,
                #                                               batch_size)
                errG = - output.mean() #+ (fake_recon_loss0 + fake_recon_loss1) * 0.2
                if epoch >= self.DAMSM_EPOCH:
                    cnn_code = image_encoder(fake_imgs)
                    s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                                 match_labels, class_ids, batch_size)
                    s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

                    errG += s_loss
                    errG.backward()
                    optimizerG.step()
                    G_logs = 'errG: %.2f s_loss: %.2f' % (
                        errG.item() - s_loss.item(), s_loss.item())
                    average_sloss.append(s_loss.item())
                    average_gloss.append(errG.item() - s_loss.item())
                    average_dloss.append(errD.item())
                else:
                    errG.backward()
                    optimizerG.step()
                    # G_logs = 'errG: %.2f Grecon: %.2f' % (errG.item(), (fake_recon_loss0 + fake_recon_loss1) * 0.2)
                    G_logs = 'errG: %.2f' % (errG.item())
                    average_sloss.append(0)
                    average_gloss.append(errG.item())
                    average_dloss.append(errD.item())
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '  ' + G_logs)

            #vutils.save_image(fake_imgs.data, '%s/fake_samples_epoch_%03d_no_average.png' % (self.image_dir, epoch),
            #                  normalize=True)

            end_t = time.time()

            print('''[%d/%d][%d] Loss_D: %.2f Loss_G: %.2f Loss_DAMSM: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     sum(average_dloss) / len(average_dloss),
                     sum(average_gloss) / len(average_gloss),
                     sum(average_sloss) / len(average_sloss),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                self.save_model(netG, avg_param_G, netD, epoch)
            torch.cuda.empty_cache()

        self.save_model(netG, avg_param_G, netD, self.max_epoch)
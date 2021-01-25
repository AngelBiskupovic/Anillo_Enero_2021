import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.optim as optim
import time
import random
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

######################################### Red Neuronal ##############################################################
class Encoder(nn.Module):
    def __init__(self, n_inputs,hidden_size=128, n_layers=1, bidirectional=False, batch_size=64, h_init=False, inversed=False,
                 xavier_init=False):
        super(Encoder, self).__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        if bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        self.GRU = nn.GRU(self.n_inputs, self.hidden_size, self.n_layers, bidirectional=bidirectional, batch_first=True, dropout=0.5)
        self.use_h_init = h_init
        self.inversed = inversed

        if self.use_h_init:
            self.h_init = nn.Parameter(
                torch.randn(self.n_layers * self.n_directions, batch_size, self.hidden_size, requires_grad=True))

        # Inicialización
        if xavier_init:
            for name, param in self.named_parameters():
               if 'bias' not in name:
                   torch.nn.init.xavier_normal_(param.data)


    def forward(self, x):

        size = x.shape
        if self.use_h_init:
            h = self.h_init[:, :size[0], :].contiguous()
        else:
            h = self.init_hidden(size[0])
        if self.inversed:
            x = x.cpu().numpy()
            x = torch.from_numpy(np.flip(x, axis=1).copy()).to(self.GRU.all_weights[0][0].device)

        self.GRU.flatten_parameters()

        x_out, h = self.GRU(x, h)
        return x_out, h

    def init_hidden(self, batch_size):
        h_init = torch.randn(self.n_layers*self.n_directions, batch_size, self.hidden_size).float().to(self.GRU.all_weights[0][0].device)
        torch.nn.init.xavier_normal_(h_init.data)
        return h_init

class Decoder(nn.Module):
    def __init__(self, n_inputs, hidden_size=128, n_layers=1, bidirectional=False, seqlen=30,h_init=False, inversed=False,
                 xavier_init=False):
        super(Decoder, self).__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.teacherForcingProb = 1.
        self.seqlen = seqlen
        self.periodDecay = 10
        if bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        self.GRU = nn.GRU(self.n_inputs, self.hidden_size, self.n_layers, bidirectional=bidirectional, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(self.n_directions*self.hidden_size, self.n_inputs)

        # Inicialización
        if xavier_init:
            for name, param in self.named_parameters():
                if 'bias' not in name:
                    torch.nn.init.xavier_normal_(param.data)


    def decayTecher(self, it):
        self.teacherForcingProb = max(0., 1. - it*(1./self.periodDecay))


    def forward(self, x, h):
        size = x.shape
        outs = []
        use_teacher_forcing = True if random.random() < self.teacherForcingProb else False
        if not self.training:
            use_teacher_forcing = False

        xout = x[:, self.seqlen:, :]
        size = xout.shape
        if use_teacher_forcing:
            x_out, h = self.GRU(xout, h)
            x_out = x_out.contiguous()
            x_out = x_out.view(size[0]*size[1], self.n_directions*self.hidden_size)
            x_out = self.linear(x_out).view(size[0], size[1], self.n_inputs)
            outs.append(x_out)

        else:
            x_out = xout[:, 0, :].view(size[0], 1, self.n_inputs)
            for i in range(xout.shape[1]):
                self.GRU.flatten_parameters()
                x_out, h = self.GRU(x_out, h)
                x_out = x_out.view(size[0]*1, self.hidden_size*self.n_directions) # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
                x_out = self.linear(x_out).view(size[0], 1, self.n_inputs)
                outs.append(x_out)
        outs = torch.cat(outs, dim=1)
        return outs


    def Predict(self, x, h, num=20):
        size = x.shape
        outs = []
        x_out = x[:, 0, :].view(size[0], 1, self.n_inputs)
        for i in range(num):
            self.GRU.flatten_parameters()
            x_out, h = self.GRU(x_out, h)
            x_out = x_out.view(size[0]*1, self.hidden_size*self.n_directions) # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
            x_out = self.linear(x_out).view(size[0], 1, self.n_inputs)
            outs.append(x_out)
        outs = torch.cat(outs, dim=1)
        return outs


    def init_hidden(self, batch_size):
        h_init = torch.randn(self.n_layers*self.n_directions, batch_size, self.hidden_size).float().to(self.GRU.all_weights[0][0].device)
        torch.nn.init.xavier_normal_(h_init.data)
        return h_init


class ModeloCama(nn.Module):
    def __init__(self, seqlen=60, preds=60, enc_inputs=4, dec_inputs=1, hidden_size=200, n_layers=1, bidirectional=False, batch_size=64,
                 h_init=False, xavier_init=False, inversed=False, weighted=False, weight_decay=0.00):
        super(ModeloCama, self).__init__()

        # Encoder
        self.weighted = weighted

        x = np.linspace(0, 1, preds)
        p1 = [0, 1]  # Puntos del tipo [pred, weight]
        p2 = [1, 3]
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        y = m * (x - p1[0]) + p1[1]
        sum_y = np.sum(y)
        self.weights = torch.from_numpy(y / sum_y).to(device).float().reshape(1, -1, 1).repeat(batch_size, 1, 1)

        self.enc_inputs = enc_inputs
        self.Encoder = Encoder(n_inputs=enc_inputs, hidden_size=hidden_size, n_layers=n_layers, bidirectional=bidirectional, batch_size=batch_size,
                               h_init=h_init, xavier_init=xavier_init, inversed=inversed).to(device)
        self.Encoder.float()

        # Parámetros decoder
        self.dec_inputs = dec_inputs
        self.Decoder = Decoder(n_inputs=dec_inputs, hidden_size=hidden_size, n_layers=n_layers, bidirectional=bidirectional, seqlen=seqlen,
                               h_init=h_init, xavier_init=xavier_init, inversed=inversed).to(device)

        self.Decoder.float()

        print('Encoder Parameters: {}'.format(self.count_trainable_parameters(self.Encoder)))
        print('Decoder Parameters: {}'.format(self.count_trainable_parameters(self.Decoder)))

        self.no_better = 0
        self.patience = 15
        self.best_val = 100000000
        self.lr = 1e-4

        # # Optimizadores
        # initial_lr = 1e-3
        # final_lr = 5e-5
        # epochs = 200
        # gamma = (1 / epochs) * np.log(final_lr / initial_lr)
        # lambda_func = lambda epoch: np.exp(gamma * epoch)


        self.encoder_optimizer = optim.Adam(self.Encoder.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.decoder_optimizer = optim.Adam(self.Decoder.parameters(), lr=self.lr, weight_decay=weight_decay)

        self.encoder_scheduler = ReduceLROnPlateau(self.encoder_optimizer, mode='min', patience=7, factor=0.2)
        self.decoder_scheduler = ReduceLROnPlateau(self.decoder_optimizer, mode='min', patience=7, factor=0.2)

        self.criterion = nn.MSELoss(reduction='sum')
        self.seqlen = seqlen
        self.preds = preds


    def decayTecher(self, it):
        self.Decoder.decayTecher(it)


    def count_trainable_parameters(self, model):
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params



    def scheduler_step(self, epoch, loss_val):
        self.encoder_scheduler.step(loss_val)
        self.decoder_scheduler.step(loss_val)



    def Train(self, Xtrain, Ytrain):
        self.Encoder.train()
        self.Decoder.train()

        Xenc = Xtrain[:, :self.seqlen, :self.enc_inputs]
        Xdec = Xtrain[:, :, -self.dec_inputs:]

        Y = Ytrain[: , self.seqlen:, :]

        # Se resetean los gradientes
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encoder-Decoder
        YencPred, h_enc = self.Encoder(Xenc)
        YdecPred = self.Decoder(Xdec, h_enc)


        # Cálculo de la pérdida
        if not self.weighted:
            loss = self.criterion(Y, YdecPred)
        else:

            loss = torch.sqrt(torch.mean(self.weights[:Y.shape[0], :, :] * torch.pow(Y - YdecPred, 2)))

        loss.backward()

        # Gradient Clipping
        for param in self.Encoder.parameters():
           param.grad.data.clamp_(-1, 1)
        for param in self.Decoder.parameters():
           param.grad.data.clamp_(-1, 1)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return float(loss.data)


    def Evaluate(self, Xtrain, Ytrain):
        with torch.no_grad():
            self.Encoder.eval()
            self.Decoder.eval()

            Xenc = Xtrain[:, :self.seqlen, :self.enc_inputs]
            Xdec = Xtrain[:, :, -self.dec_inputs:]

            Y = Ytrain[:, self.seqlen:, :]

            # Encoder-Decoder
            YencPred, h_enc = self.Encoder(Xenc)
            YdecPred = self.Decoder(Xdec, h_enc)

            # Cálculo de la pérdida
            if not self.weighted:
                loss = self.criterion(Y, YdecPred)
            else:

                loss = torch.sqrt(torch.mean(self.weights[:Y.shape[0], :, :] * torch.pow(Y - YdecPred, 2)))

            return float(loss.data)


    def Predict(self, Xtest, num=20):
        with torch.no_grad():
            self.Encoder.eval()
            self.Decoder.eval()

            Xenc = Xtest[:, :self.seqlen, :self.enc_inputs]
            Xdec = Xtest[:, self.seqlen:, -self.dec_inputs:]

            # Encoder-Decoder
            YencPred, h_enc = self.Encoder(Xenc)
            YdecPred = self.Decoder.Predict(Xdec, h_enc, num)

            return YdecPred

    def save(self, folder, name):
        torch.save(self.Encoder.state_dict(), folder + 'encoder' + name)
        torch.save(self.Decoder.state_dict(), folder + 'decoder' + name)


    def load(self, folder, name):
        try:
            self.Encoder.load_state_dict(torch.load(folder + 'encoder' + name))
            self.Decoder.load_state_dict(torch.load(folder + 'decoder' + name))
            print('Modelos cargados con éxito')
        except:
            print('Modelos no se han podido cargar')

    def load_state_dict(self, state_dict, strict=True):
        self.Encoder.load_state_dict(state_dict[0])
        self.Decoder.load_state_dict(state_dict[1])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.Encoder.state_dict(), self.Decoder.state_dict()





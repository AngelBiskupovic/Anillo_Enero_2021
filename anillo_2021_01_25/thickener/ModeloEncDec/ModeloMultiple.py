import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import os
from sklearn.preprocessing import StandardScaler
from ModeloEncDec.ModeloUnico import ModeloEncDec as ModeloEncDecUnico


class ModeloEncDec():
    def __init__(self, bed_variables, pressure_variables, torque_variables, solidC_variables, batch_size=16,
                 preds=60, seqlen=60, parameters=[], weighted=False, h_init=False, inversed=False,
                 hidden_size=75, folder='Nets/ModeloRecurrente/', xavier_init=False, model_type='_pre'):

        self.preds = preds
        self.seqlen = seqlen
        self.parameters = parameters
        self.weighted = weighted
        self.h_init = h_init
        self.inversed = inversed
        self.hidden_size = hidden_size
        self.folder = folder
        self.xavier_init = xavier_init
        self.batch_size = batch_size
        self.bed_variables = bed_variables
        self.pressure_variables = pressure_variables
        self.torque_variables = torque_variables
        self.solidC_variables = solidC_variables


        var = 'bed'
        self.bed = ModeloEncDecUnico(bed_variables[0], bed_variables[1], model=var,
                                        batch_size=batch_size, preds=preds, seqlen=seqlen, parametros=parameters,
                                        recuperado='best_{}_model{}.pt'.format(var, model_type),
                                        saveName='{}_model.pt'.format(var),
                                        weighted=weighted,
                                        h_init= h_init,
                                        inversed=inversed,
                                        hidden_size=hidden_size,
                                        folder=folder + '/{}/'.format(var),
                                        xavier_init=xavier_init)

        var = 'pressure'
        self.pressure = ModeloEncDecUnico(pressure_variables[0], pressure_variables[1], model=var,
                                         batch_size=batch_size, preds=preds, seqlen=seqlen, parametros=parameters,
                                         recuperado='best_{}_model{}.pt'.format(var, model_type),
                                         saveName='{}_model.pt'.format(var),
                                         weighted=weighted,
                                         h_init=h_init,
                                         inversed=inversed,
                                         hidden_size=hidden_size,
                                         folder=folder + '/{}/'.format(var),
                                         xavier_init=xavier_init)

        var = 'torque'
        self.torque = ModeloEncDecUnico(torque_variables[0], torque_variables[1], model=var,
                                              batch_size=batch_size, preds=preds, seqlen=seqlen, parametros=parameters,
                                              recuperado='best_{}_model{}.pt'.format(var, model_type),
                                              saveName='{}_model.pt'.format(var),
                                              weighted=weighted,
                                              h_init=h_init,
                                              inversed=inversed,
                                              hidden_size=hidden_size,
                                              folder=folder + '/{}/'.format(var),
                                              xavier_init=xavier_init)

        var = 'solidC'
        self.solidC = ModeloEncDecUnico(solidC_variables[0], solidC_variables[1], model=var,
                                            batch_size=batch_size, preds=preds, seqlen=seqlen, parametros=parameters,
                                            recuperado='best_{}_model{}.pt'.format(var, model_type),
                                            saveName='{}_model.pt'.format(var),
                                            weighted=weighted,
                                            h_init=h_init,
                                            inversed=inversed,
                                            hidden_size=hidden_size,
                                            folder=folder + '/{}/'.format(var),
                                            xavier_init=xavier_init)

        self.weights = self.bed.weights


    def set_learning_rate(self, lr):
        self.bed.set_learning_rate(lr=lr)
        self.pressure.set_learning_rate(lr=lr)
        self.torque.set_learning_rate(lr=lr)
        self.solidC.set_learning_rate(lr=lr)

    def decay_teacher_prob(self, it):
        self.bed.decay_teacher_prob(it)
        self.pressure.decay_teacher_prob(it)
        self.torque.decay_teacher_prob(it)
        self.solidC.decay_teacher_prob(it)

    def scheduler_step(self, epoch, eval_loss):
        self.bed.scheduler.step(epoch, eval_loss)
        self.pressure.scheduler.step(epoch, eval_loss)
        self.torque.scheduler.step(epoch, eval_loss)
        self.solidC.scheduler.step(epoch, eval_loss)


    def train(self, Xy_bed, Xy_pressure, Xy_torque, Xy_solidC, still_training=[0, 1, 2, 3]):

        if 0 in still_training:
            bed_loss = self.bed.train(Xy_bed)
        else:
            bed_loss = -1

        if 1 in still_training:
            pressure_loss = self.pressure.train(Xy_pressure)
        else:
            pressure_loss = -1

        if 2 in still_training:
            torque_loss = self.torque.train(Xy_torque)
        else:
            torque_loss = -1


        if 3 in still_training:
            solidC_loss = self.solidC.train(Xy_solidC)
        else:
            solidC_loss = -1


        return bed_loss, pressure_loss, torque_loss, solidC_loss


    def evaluate(self, Xy_bed, Xy_pressure, Xy_torque, Xy_solidC, still_training=[0, 1, 2, 3]):

        if 0 in still_training:
            bed_loss = self.bed.evaluate(Xy_bed)
        else:
            bed_loss = -1

        if 1 in still_training:
            pressure_loss = self.pressure.evaluate(Xy_pressure)
        else:
            pressure_loss = -1

        if 2 in still_training:
            torque_loss = self.torque.evaluate(Xy_torque)
        else:
            torque_loss = -1

        if 3 in still_training:
            solidC_loss = self.solidC.evaluate(Xy_solidC)
        else:
            solidC_loss = -1

        return bed_loss, pressure_loss, torque_loss, solidC_loss


    def predict(self, Xy_bed, Xy_pressure, Xy_torque, Xy_solidC, num=60, denormalice=True, cpu=True, reshape=True, still_predict=[0, 1, 2, 3]):
        if 0 in still_predict:
            bed_pred = self.bed.predict(Xy_bed, num=num, denormalice=denormalice, cpu=cpu, reshape=reshape)
        else:
            bed_pred = [-1]
        if 1 in still_predict:
            pressure_pred = self.pressure.predict(Xy_pressure, num=num, denormalice=denormalice, cpu=cpu, reshape=reshape)
        else:
            pressure_pred = [-1]
        if 2 in still_predict:
            torque_pred = self.torque.predict(Xy_torque, num=num, denormalice=denormalice, cpu=cpu, reshape=reshape)
        else:
            torque_pred = [-1]
        if 3 in still_predict:
            solidC_pred = self.solidC.predict(Xy_solidC, num=num, denormalice=denormalice, cpu=cpu, reshape=reshape)
        else:
            solidC_pred = [-1]

        return bed_pred, pressure_pred, torque_pred, solidC_pred


    def save(self, type='norm', index=[0, 1, 2, 3]):
        models = [self.bed, self.pressure, self.torque, self.solidC]
        for i in index:
            models[i].save(type=type)


    def load(self, index=[0, 1, 2, 3]):
        models = [self.bed, self.pressure, self.torque, self.solidC]
        for i in index:
            models[i].load()


    def load_state_dict(self, state_dicts, indexes=[0, 1, 2, 3]):
        models = [self.bed, self.pressure, self.torque, self.solidC]
        for ind in indexes:
            models[ind].load_state_dict(state_dicts[ind])

    def state_dict(self):
        return self.bed.state_dict(), self.pressure.state_dict(), self.torque.state_dict(), self.solidC.state_dict()


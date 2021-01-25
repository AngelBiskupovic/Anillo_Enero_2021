import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.optim as optim
import random
from sklearn.preprocessing import StandardScaler
from collections import deque
from ModeloEncDec.ModeloCama import ModeloCama


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModeloEncDec():
    def __init__(self, Inputs, Outputs, model='bed',
                 batch_size=128, preds=30, seqlen=50, parametros=[], recuperado='', saveName='', folder='Redes/', xavier_init=False,
                 h_init=False, inversed=False, weighted=False, hidden_size=100, n_layers=1, weight_decay=0.0):

        self.folder = folder
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.preds = preds
        self.evalLossMinCama = 100000
        self.saveName = saveName


        self.batch = deque(maxlen=batch_size)
        self.model_name = model

        self.model = ModeloCama(seqlen=seqlen, preds=preds, enc_inputs=len(Inputs), dec_inputs=len(Outputs), hidden_size=hidden_size, n_layers=n_layers, bidirectional=False, batch_size=batch_size,
                                xavier_init=xavier_init, h_init=h_init, inversed=inversed, weighted=weighted, weight_decay=weight_decay).to(device)

        self.weights = self.model.weights
        self.Inputs = Inputs
        self.Outputs = Outputs

        self.AllTags = self.Inputs
        self.recuperado = recuperado
        self.load()

        self.parametros = parametros

        self.Centros = {}#np.array([parametros[x]['center'] for x in parametros.keys() if x != 'epoch' and x != 'same']).reshape(1, -1)
        self.Escalas = {}#np.array([parametros[x]['scale'] for x in parametros.keys() if x != 'epoch' and x != 'same']).reshape(1,-1)
        for key in self.parametros.keys():
            if key != 'epoch' and key != 'same':
                self.Centros[key] = [self.parametros[key]['center']]
                self.Escalas[key] = [self.parametros[key]['scale']]

        self.Centros = pd.DataFrame(self.Centros)
        self.Escalas = pd.DataFrame(self.Escalas)


    def set_learning_rate(self, lr):
        # Optimizadores
        self.model.optimizer = optim.Adam(list(self.model.Encoder.parameters()) + list(self.model.Decoder.parameters()), lr=lr)


    def decay_teacher_prob(self, it):
        self.model.decayTecher(it)


    def scheduler_step(self, epoch, val_loss):
        self.model.scheduler_step(epoch, val_loss)


    def normalice(self, X):
        columns = X.columns
        X = (np.array(X) - np.array(self.Centros[columns]))/np.array(self.Escalas[columns])
        return pd.DataFrame(X, columns=columns)

    def train(self, Xy):
        X = Xy[0]
        Y = Xy[1]

        loss = self.model.Train(X, Y)
        return loss


    def evaluate(self, Xy):

        X = Xy[0]
        Y = Xy[1]

        self.model.to(torch.device('cpu'))

        loss = self.model.Evaluate(X, Y)

        self.model.to(device)

        return loss


    def predict(self, Xy, num=20, denormalice=True, cpu=True, reshape=True):
        X = Xy

        if cpu:
            self.model.to(torch.device('cpu'))

        # Se realizan las predicciones
        if denormalice:
            if cpu:
                YPred = self.model.Predict(X, num).numpy().squeeze()*self.parametros[self.Outputs[0]]['scale'] + self.parametros[self.Outputs[0]]['center']
            else:
                YPred = self.model.Predict(X, num).squeeze()*self.parametros[self.Outputs[0]]['scale'] + self.parametros[self.Outputs[0]]['center']

        else:
            if cpu:
                YPred = self.model.Predict(X, num).numpy().squeeze()
            else:
                YPred = self.model.Predict(X, num).squeeze()

        if cpu:
            self.model.to(device)


        if reshape:
            return YPred.reshape(-1, 1)
        else:
            return YPred


    def save(self, type='norm'):
        if type != 'norm':
            torch.save(self.model.state_dict(), self.folder + type + '_' + self.saveName)
        else:
            torch.save(self.model.state_dict(), self.folder + self.saveName)



    def load(self):
        try:
            if self.recuperado != '':
                self.model.load_state_dict(torch.load(self.folder + self.recuperado, map_location=device))
                print('Modelos cargados con éxito')
            else:
                print('No se han podido cargar los modelos')
        except Exception as inst:
            print('No se han podido cargar el modelos')
            print(inst)



    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()



if __name__ == "__main__":

    tagsDict = {'e_7110_dt_1011_solido': 'C. Sólidos Entrada',  'e_7120_ft_1002':'F. Floculante',
                'e_7120_ft_1001': 'F. Agua Dilución', 'e_7110_ft_1012': 'F. Entrada', 'e_7110_ot_1003': 'Torque Rastrillo',
                  'e_7110_lt_1009_s4': 'Cama', 'e_7110_lt_1009_s3': 'Barro', 'e_7110_lt_1009': 'Claridad',
                'e_7110_dt_1030_solido': 'C. Sólidos Salida', 'e_7110_dt_1011': 'D. Sólidos Entrada', 'e_7110_dt_1030': 'D. Sólidos Salida',
                'e_7110_ft_1030': 'F. Salida', 'e_7110_lt_1009_s2': 'Interfase', 'e_7110_pt_1010': 'P. Hidroestática',
                'e_7120_lit_001': 'N. Estanque Floculante'}

    # Inputs Cama
    CamaInputs = ['e_7120_ft_1002', 'e_7110_ft_1012', 'e_7110_dt_1011_solido', 'e_7110_lt_1009_s4']
    CamaOutputs = ['e_7110_lt_1009_s4']

    # Inputs Presión
    PresionInputs = ['e_7120_ft_1002', 'e_7110_ft_1012', 'e_7110_dt_1011_solido', 'e_7110_pt_1010']
    PresionOutputs = ['e_7110_pt_1010']

    # Inputs CSólidos
    CSolidosInputs = ['e_7110_pt_1010', 'e_7110_ft_1030', 'e_7110_dt_1030_solido']
    CSolidosOutputs = ['e_7110_dt_1030_solido']


    with open('../parametros.pkl', 'rb') as file:
        parametros = pickle.load(file)

    def Save(obj, name):
        with open(name, 'wb') as file:
            pickle.dump(obj, file)
    def Load(name):
        with open(name, 'rb') as file:
            return pickle.load(file)



    data = pd.read_pickle('../todos_5minPreproc.pkl')


    modelo = ModeloEncDecAttn(CamaInputs, CamaOutputs, PresionInputs,PresionOutputs, CSolidosInputs, CSolidosOutputs,
                 batch_size=128, preds=20, seqlen=50, parametros=parametros,
                              recuperados=['CamaPre.pt', 'PresionPre.pt', 'CSolidosPre.pt'],
                              saveNames=['CamaEncDecAttn.pt', 'PresionEncDecAttn.pt', 'CSolidosEncDecAttn.pt'], folder='Redes/')

    for i in range(len(data)):
        X = data.iloc[i: i + 70, :]
        trains = modelo.ArrangeData(X, 'train')
        preds = modelo.Train(trains[0], trains[1], trains[2])
        print(preds)



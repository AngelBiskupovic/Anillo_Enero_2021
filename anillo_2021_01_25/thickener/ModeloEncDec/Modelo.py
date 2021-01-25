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
from ModeloEncDecSimple.ModeloPresion import ModeloPresion
from ModeloEncDecSimple.ModeloCSolidos import ModeloCSolidos
from ModeloEncDecSimple.ModeloCama import ModeloCama

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModeloEncDecSimple():
    def __init__(self, CamaInputs, CamaOutputs, PresionInputs,PresionOutputs, CSolidosInputs, CSolidosOuputs,
                 batch_size=128, preds=30, seqlen=30, parametros=[], recuperados=[], saveNames=[], folder='Redes/'):
        self.folder= folder
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.preds = preds
        self.evalLossMinCama = 100000
        self.evalLossMinCSolidos = 10000
        self.evalLossMinPresion = 10000
        self.saveNames = saveNames

        self.batchCama = deque(maxlen=batch_size)
        self.batchPresion = deque(maxlen=batch_size)
        self.batchCsolidos = deque(maxlen=batch_size)

        # ModeloCama
        self.Cama = ModeloCama(seqlen=seqlen, enc_inputs=4, dec_inputs=1, hidden_size=100, n_layers=1, bidirectional=False, batch_size=batch_size).to(device)
        self.CamaInputs = CamaInputs
        self.CamaOutputs = CamaOutputs

        # ModeloPresion
        self.Presion = ModeloPresion(seqlen=seqlen, enc_inputs=4, dec_inputs=1, hidden_size=100, n_layers=1, bidirectional=False, batch_size=batch_size).to(device)
        self.PresionInputs = PresionInputs
        self.PresionOutputs = PresionOutputs

        # CSolidos
        self.CSolidos = ModeloCSolidos(seqlen=seqlen, enc_inputs=4, dec_inputs=1, hidden_size=100, n_layers=1, bidirectional=False, batch_size=batch_size).to(device)
        self.CSolidosInputs = CSolidosInputs
        self.CSolidosOutputs = CSolidosOuputs

        self.AllTags = list(set(self.CamaInputs + self.CSolidosInputs + self.PresionInputs))
        self.recuperados = recuperados
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
        self.Cama.Enc_optimizer = optim.Adam(self.Cama.Encoder.parameters(), lr=lr)
        self.Cama.Dec_optimizer = optim.Adam(self.Cama.Decoder.parameters(), lr=lr)

        self.Presion.Enc_optimizer = optim.Adam(self.Presion.Encoder.parameters(), lr=lr)
        self.Presion.Dec_optimizer = optim.Adam(self.Presion.Decoder.parameters(), lr=lr)

        self.CSolidos.Enc_optimizer = optim.Adam(self.CSolidos.Encoder.parameters(), lr=lr)
        self.CSolidos.Dec_optimizer = optim.Adam(self.CSolidos.Decoder.parameters(), lr=lr)

    def decay_teacher_prob(self, it):
        self.Cama.decayTecher(it)
        self.Presion.decayTecher(it)
        self.CSolidos.decayTecher(it)


    def normalice(self, X):
        columns = X.columns
        X = (np.array(X) - np.array(self.Centros[columns]))/np.array(self.Escalas[columns])
        return pd.DataFrame(X, columns=columns)

    def train(self, XyCama, XyPresion, XySolidos, still_training=[0, 1, 2]):
        XCama = XyCama[0]
        YCama = XyCama[1]
        XPresion = XyPresion[0]
        YPresion = XyPresion[1]
        XSolidos = XySolidos[0]
        YSolidos = XySolidos[1]

        if 0 in still_training:
            lossCama = self.Cama.Train(XCama, YCama)
        else:
            lossCama = -1
        if 1 in still_training:
            lossPresion = self.Presion.Train(XPresion, YPresion)
        else:
            lossPresion = -1
        if 2 in still_training:
            lossSolidos = self.CSolidos.Train(XSolidos, YSolidos)
        else:
            lossSolidos = -1
        return lossCama, lossPresion, lossSolidos


    def evaluate(self, XyCama, XyPresion, XySolidos, still_training=[0, 1, 2]):

        XCama = XyCama[0]
        YCama = XyCama[1]
        XPresion = XyPresion[0]
        YPresion = XyPresion[1]
        XSolidos = XySolidos[0]
        YSolidos = XySolidos[1]

        if 0 in still_training:
            self.Cama.to(torch.device('cpu'))
            lossCama = self.Cama.Evaluate(XCama, YCama)
            self.Cama.to(device)
        else:
            lossCama = - 1
        if 1 in still_training:
            self.Presion.to(torch.device('cpu'))
            lossPresion = self.Presion.Evaluate(XPresion, YPresion)
            self.Presion.to(device)
        else:
            lossPresion = -1
        if 2 in still_training:
            self.CSolidos.to(torch.device('cpu'))
            lossSolidos = self.CSolidos.Evaluate(XSolidos, YSolidos)
            self.CSolidos.to(device)
        else:
            lossSolidos = -1

        return lossCama, lossPresion , lossSolidos


    def predict(self, XyCama, XyPresion, XySolidos, num=20, denormalice=True):
        XCama = XyCama
        XPresion = XyPresion
        XSolidos = XySolidos

        self.Cama.to(torch.device('cpu'))
        self.Presion.to(torch.device('cpu'))
        self.CSolidos.to(torch.device('cpu'))

        # Se realizan las predicciones
        if denormalice:
            YCamaPred = self.Cama.Predict(XCama, num).numpy().squeeze() * self.parametros[self.CamaOutputs[0]][
                'scale'] + self.parametros[self.CamaOutputs[0]]['center']
            YPresionPred = self.Presion.Predict(XPresion, num).numpy().squeeze() * \
                           self.parametros[self.PresionOutputs[0]]['scale'] + self.parametros[self.PresionOutputs[0]][
                               'center']
            YSolidosPred = self.CSolidos.Predict(XSolidos, num).numpy().squeeze() * \
                           self.parametros[self.CSolidosOutputs[0]]['scale'] + self.parametros[self.CSolidosOutputs[0]][
                               'center']

        else:
            YCamaPred = self.Cama.Predict(XCama, num).numpy().squeeze()
            YPresionPred = self.Presion.Predict(XPresion, num).numpy().squeeze()
            YSolidosPred = self.CSolidos.Predict(XSolidos, num).numpy().squeeze()


        self.Cama.to(device)
        self.Presion.to(device)
        self.CSolidos.to(device)

        return YCamaPred.reshape(-1, 1), YPresionPred.reshape(-1, 1), YSolidosPred.reshape(-1, 1)

    def arrange_data_train(self, X, batch):
        X = X.reshape(1, self.seqlen + self.preds, -1)
        batch.append(X)

        local_batch = np.concatenate(list(batch), axis=0)
        Xtrain = local_batch[:, :, :-1]
        Ytrain = local_batch[:, :, -1:]
        Xtrain = torch.from_numpy(Xtrain).float().to(device)
        Ytrain = torch.from_numpy(Ytrain).float().to(device)
        return Xtrain, Ytrain

    def arrange_data_eva(self, X):
        X = X.reshape(1, self.seqlen + self.preds, -1)
        Xtrain = X[:, :self.seqlen + self.preds, :-1]
        Ytrain = X[:, :, -1:].reshape(1, self.preds + self.seqlen, -1)
        Xtrain = torch.from_numpy(Xtrain).float().to(device)
        Ytrain = torch.from_numpy(Ytrain).float().to(device)
        return Xtrain, Ytrain

    def arrange_data_predict(self, X):
        X = X[-self.seqlen:, :]
        X = X.reshape(1, self.seqlen , -1)
        return (torch.from_numpy(X[:, :, :-1]).float().to(device), 0)

    def arrange_data(self, data, format='train'): # Le llegan secuenciasda
        data = self.Normalice(data[self.AllTags])
        XCama = np.array(pd.concat([data[self.CamaInputs], data[self.CamaOutputs]], axis=1)) # Nos aseguramos que están en el orden correcto.
        XPresion = np.array(pd.concat([data[self.PresionInputs], data[self.PresionOutputs]], axis=1)) # Nos aseguramos que están en el orden correcto.
        XCSolidos = np.array(pd.concat([data[self.CSolidosInputs], data[self.CSolidosOutputs]], axis=1)) # Nos aseguramos que están en el orden correcto.

        if format == 'train':
            XyCama = self.ArrangeDataTrain(XCama, self.batchCama)
            XyPresion = self.ArrangeDataTrain(XPresion, self.batchPresion)
            XySolidos = self.ArrangeDataTrain(XCSolidos, self.batchCsolidos)
            return XyCama, XyPresion, XySolidos

        elif format == 'eval':
            XyCama = self.ArrangeDataEval(XCama)
            XyPresion = self.ArrangeDataEval(XPresion)
            XySolidos = self.ArrangeDataEval(XCSolidos)
            return XyCama, XyPresion, XySolidos


        else:
            XyCama = self.ArrangeDataPredict(XCama)
            XyPresion = self.ArrangeDataPredict(XPresion)
            XySolidos = self.ArrangeDataPredict(XCSolidos)
            return XyCama, XyPresion, XySolidos


    def save(self, type='norm', index=[0,1,2]):
        state_dicts = [self.Cama.state_dict(), self.Presion.state_dict(), self.CSolidos.state_dict()]
        if self.saveNames == []:
            if type != 'norm':
                torch.save(self.Cama.state_dict(), self.folder + type + '_' + 'camaEncDec.pt')
                torch.save(self.Presion.state_dict(), self.folder + type + '_' +'presionEncDec.pt')
                torch.save(self.CSolidos.state_dict(), self.folder + type + '_' + 'CsolidosEncDec.pt')
            else:
                torch.save(self.Cama.state_dict(), self.folder + 'camaEncDec.pt')
                torch.save(self.Presion.state_dict(), self.folder + 'presionEncDec.pt')
                torch.save(self.CSolidos.state_dict(), self.folder + 'CsolidosEncDec.pt')
        else:
            for ind in index:
                if type == 'norm':
                    torch.save(state_dicts[ind], self.folder + self.saveNames[ind])
                else:
                    torch.save(state_dicts[ind], self.folder + type + '_' + self.saveNames[ind])



    def load(self):
        try:
            if self.recuperados != []:
                self.Cama.load_state_dict(torch.load(self.folder + self.recuperados[0]))
                self.Presion.load_state_dict(torch.load(self.folder + self.recuperados[1]))
                self.CSolidos.load_state_dict(torch.load(self.folder + self.recuperados[2]))
                print('Modelos cargados con éxito')
            else:
                print('No se han podido cargar los modelos')
        except Exception as inst:
            print(inst)
            print('No se han podido cargar el modelos')



    def load_state_dict(self, state_dicts, indexes=[0, 1, 2]):
        models = [self.Cama, self.Presion, self.CSolidos]
        for ind in indexes:
            models[ind].load_state_dict(state_dicts[ind])

    def state_dict(self):
        return self.Cama.state_dict(), self.Presion.state_dict(), self.CSolidos.state_dict()


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


    modelo = ModeloEncDecSimple(CamaInputs, CamaOutputs, PresionInputs,PresionOutputs, CSolidosInputs, CSolidosOutputs,
                 batch_size=128, preds=20, seqlen=30, parametros=parametros,
                              recuperados=['CamaPre.pt', 'PresionPre.pt', 'CSolidosPre.pt'],
                              saveNames=['CamaEncDecSimple.pt', 'PresionEncDecSimple.pt', 'CSolidosEncDecSimple.pt'], folder='Redes/')

    for i in range(len(data)):
        X = data.iloc[i: i + 50, :]
        trains = modelo.ArrangeData(X, 'train')
        preds = modelo.Tr(trains[0], trains[1], trains[2])
        print(preds)



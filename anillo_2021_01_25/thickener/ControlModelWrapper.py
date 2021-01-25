import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import datetime
import copy
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ControlWrapper:
    def __init__(self, model, scaler, scaler_dict):
        self.model = model
        self.scaler = scaler
        self.scaler_dict = scaler_dict
        self.seqlen = self.model.seqlen
        self.preds = self.model.preds
        self.bed_array = []
        self.pressure_array = []
        self.torque_array = []
        self.solidC_array = []
        self.bed_indexes = {self.model.bed_variables[0][i]: i for i in range(len(self.model.bed_variables[0]))}
        self.pressure_indexes = {self.model.pressure_variables[0][i]: i for i in range(len(self.model.pressure_variables[0]))}
        self.torque_indexes = {self.model.torque_variables[0][i]: i for i in range(len(self.model.torque_variables[0]))}
        self.solidC_indexes = {self.model.solidC_variables[0][i]: i for i in range(len(self.model.solidC_variables[0]))}
        self.t = 0



    def set_initial_conditions(self, initial_dict, batch_size=1, scaled_input=False, to_torch=True, array=False):
        '''
        Se crean las matrices iniciales para los modelos
        :param initial_dict: diccionario cuyas llaves son los tags y los valores son las condiciones iniciales
        :param batch_size:
        :param scaled_input: Indica si las entradas están escaladas a media cero y varianza 1  o no
        :param array: Si es un diccionario con arreglos o un diccionario con floats
        :param scaled_output: Si se entrega el output con media cero y varianza 1 o no
        Por defecto se asume que las entradas y las salidas no vienen escaladas, es decir, que están en sus dimensiones originales.
        :return:
        '''
        # Se escalan las entradas en caso de que no lo estén
        if not scaled_input:
            for key in initial_dict.keys():
                mean = self.scaler_dict[key]['mean']
                std = self.scaler_dict[key]['std']
                initial_dict[key] = (initial_dict[key] - mean)/std


        # Se guardan las entradas por cada modelo
        bed_variables = self.model.bed_variables[0]
        pressure_variables = self.model.pressure_variables[0]
        torque_variables = self.model.torque_variables[0]
        solidC_variables = self.model.solidC_variables[0]

        if not array:
            bed_variables = np.array([initial_dict[var] for var in bed_variables]).reshape(1, 1, -1)
            pressure_variables = np.array([initial_dict[var] for var in pressure_variables]).reshape(1, 1, -1)
            torque_variables = np.array([initial_dict[var] for var in torque_variables]).reshape(1, 1, -1)
            solidC_variables = np.array([initial_dict[var] for var in solidC_variables]).reshape(1, 1, -1)

            # Se les da las dimensiones necesarias
            bed_variables = np.repeat(np.repeat(bed_variables, repeats=self.seqlen + 1, axis=1), repeats=batch_size, axis=0)
            pressure_variables = np.repeat(np.repeat(pressure_variables, repeats=self.seqlen + 1, axis=1), repeats=batch_size, axis=0)
            torque_variables = np.repeat(np.repeat(torque_variables, repeats=self.seqlen + 1, axis=1), repeats=batch_size, axis=0)
            solidC_variables = np.repeat(np.repeat(solidC_variables, repeats=self.seqlen + 1, axis=1), repeats=batch_size, axis=0)

        else:
            bed_variables = np.repeat(np.array([initial_dict[var] for var in bed_variables]).reshape(1, self.seqlen + 1, -1), repeats=batch_size, axis=0)
            pressure_variables = np.repeat(np.array([initial_dict[var] for var in pressure_variables]).reshape(1, self.seqlen + 1, -1), repeats=batch_size, axis=0)
            torque_variables = np.repeat(np.array([initial_dict[var] for var in torque_variables]).reshape(1, self.seqlen + 1, -1), repeats=batch_size, axis=0)
            solidC_variables = np.repeat(np.array([initial_dict[var] for var in solidC_variables]).reshape(1, self.seqlen + 1, -1), repeats=batch_size, axis=0)


        if to_torch:
            bed_variables = torch.from_numpy(bed_variables)
            pressure_variables = torch.from_numpy(pressure_variables)
            torque_variables = torch.from_numpy(torque_variables)
            solidC_variables = torch.from_numpy(solidC_variables)

        self.bed_array = bed_variables
        self.pressure_array = pressure_variables
        self.torque_array = torque_variables
        self.solidC_array = solidC_variables

        return bed_variables, pressure_variables, torque_variables, solidC_variables


    def modify_array(self, array, input_dict, indexes):
        use_indexes = []
        values = []
        # Se extraen los índices deseados
        for key in input_dict.keys():
            try:
                use_indexes.append(indexes[key])
                values.append(torch.from_numpy(input_dict[key]).float().reshape(array.shape[0], 1, 1))
            except KeyError:
                pass

        values = torch.cat(values, dim=2)
        aux = copy.deepcopy(array[:, -1:, :])
        aux[:, :, use_indexes] = values
        array = torch.cat([array[:, 1:, :], aux], dim=1)
        return array


    def run(self, u_dict_original, scaled_input=False, cascaded=True, scaled_output=False):

        # Escalamiento de las entradas en caso de que vengan en sus dimensiones originales
        u_dict = copy.deepcopy(u_dict_original)
        if not scaled_input:
            for key in u_dict.keys():
                mean = self.scaler_dict[key]['mean']
                std = self.scaler_dict[key]['std']
                u_dict[key] = (u_dict[key] - mean)/std

        # Se modifican los arreglos de entrada con las nuevas entradas
        self.bed_array = self.modify_array(self.bed_array, u_dict, indexes=self.bed_indexes)
        self.pressure_array = self.modify_array(self.pressure_array, u_dict, indexes=self.pressure_indexes)
        self.torque_array = self.modify_array(self.torque_array, u_dict, indexes=self.torque_indexes)
        self.solidC_array = self.modify_array(self.solidC_array, u_dict, indexes=self.solidC_indexes)

        # Realización de las predicciones
        bed_pred, pressure_pred, torque_pred, solidC_pred = \
            self.model.predict(self.bed_array, self.pressure_array, self.torque_array, self.solidC_array, num=1, denormalice=False, cpu=True,
                                reshape=False)

        self.bed_array = self.modify_array(self.bed_array, {self.model.bed_variables[1][0]: bed_pred}, indexes=self.bed_indexes)
        self.pressure_array = self.modify_array(self.pressure_array, {self.model.pressure_variables[1][0]: pressure_pred}, indexes=self.pressure_indexes)
        self.torque_array = self.modify_array(self.torque_array, {self.model.torque_variables[1][0]: torque_pred}, indexes=self.torque_indexes)
        if not cascaded:
            self.solidC_array = self.modify_array(self.solidC_array, {self.model.solidC_variables[1][0]: solidC_pred}, indexes=self.solidC_indexes)
        if cascaded:
            self.solidC_array = self.modify_array(self.solidC_array, {self.model.solidC_variables[1][0]: solidC_pred,
                                                                      self.model.bed_variables[1][0]: bed_pred,
                                                                      self.model.pressure_variables[1][0]: pressure_pred,
                                                                      self.model.torque_variables[1][0]: torque_pred}, indexes=self.solidC_indexes)

        if not scaled_output:
            bed_pred = bed_pred*self.scaler_dict[self.model.bed_variables[1][0]]['std'] + self.scaler_dict[self.model.bed_variables[1][0]]['mean']
            pressure_pred = pressure_pred*self.scaler_dict[self.model.pressure_variables[1][0]]['std'] + self.scaler_dict[self.model.pressure_variables[1][0]]['mean']
            torque_pred = torque_pred*self.scaler_dict[self.model.torque_variables[1][0]]['std'] + self.scaler_dict[self.model.torque_variables[1][0]]['mean']
            solidC_pred = solidC_pred*self.scaler_dict[self.model.solidC_variables[1][0]]['std'] + self.scaler_dict[self.model.solidC_variables[1][0]]['mean']

        out = np.concatenate([bed_pred.reshape(self.bed_array.shape[0], 1, 1), pressure_pred.reshape(self.bed_array.shape[0], 1, 1),
                         torque_pred.reshape(self.bed_array.shape[0], 1, 1), solidC_pred.reshape(self.bed_array.shape[0], 1, 1)], axis=2)

        return out




if __name__ == '__main__':
    from Nets.ModeloRecurrente.ModeloMultiple import ModeloRecurrente
    from Nets.ModeloEncDec.ModeloMultiple import ModeloEncDec
    from Nets.ModeloEncDecAttn.ModeloMultiple import ModeloEncDecAttn
    from Nets.ModeloEncDecMultipleHead.ModeloMultiple import ModeloEncDecMultipleHead
    from Nets.ModeloEncDecSimpleRL.ModeloMultiple import ModeloEncDecSimpleRL
    from Nets.ModeloRecurrenteRL.ModeloMultiple import ModeloRecurrenteRL
    from Nets.ModeloFutureDependentVAEMLP.ModeloMultiple import ModeloFutureDependentVAEMLP
    from Nets.ModeloEncDecDoubleAttn.ModeloMultiple import ModeloEncDecDoubleAttn


    def count_trainable_parameters(model):
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params

    def save(obj, name):
        with open(name, 'wb') as file:
            pickle.dump(obj, file)


    def load(name):
        with open(name, 'rb') as file:
            return pickle.load(file)


    with open('DataFiles/parametrosNew.pkl', 'rb') as file:
        parameters = pickle.load(file)

    # Inputs Cama
    bed_inputs = ['br_7120_ft_1002', 'bj_7110_ft_1012', 'bg_7110_dt_1011_solido', 'bo_7110_lt_1009_s4']
    bed_u = ['br_7120_ft_1002']
    bed_outputs = ['bo_7110_lt_1009_s4']

    # Inputs Presión
    pressure_inputs = ['br_7120_ft_1002', 'bj_7110_ft_1012', 'bg_7110_dt_1011_solido', 'bq_7110_pt_1010']
    pressure_u = ['br_7120_ft_1002']
    pressure_outputs = ['bq_7110_pt_1010']

    # Inputs Presión
    torque_inputs = ['br_7120_ft_1002', 'bj_7110_ft_1012', 'bg_7110_dt_1011_solido', 'bp_7110_ot_1003']
    torque_u = ['br_7120_ft_1002']
    torque_outputs = ['bp_7110_ot_1003']

    # Inputs CSólidos
    solidC_inputs = ['br_7120_ft_1002', 'bk_7110_ft_1030', 'bq_7110_pt_1010', 'bo_7110_lt_1009_s4',
                     'bp_7110_ot_1003', 'bi_7110_dt_1030_solido']

    solidC_u = ['bk_7110_ft_1030']
    solidC_outputs = ['bi_7110_dt_1030_solido']

    bed_variables = [bed_inputs, bed_outputs]
    pressure_variables = [pressure_inputs, pressure_outputs]
    torque_variables = [torque_inputs, torque_outputs]
    solidC_variables = [solidC_inputs, solidC_outputs]

    # Scaler
    scaler = torch.load('DataFiles/new_data_all.pkl')['scaler']
    data = torch.load('DataFiles/new_data_all.pkl')['new_data']
    columns = data.columns
    data = scaler.inverse_transform(np.array(data))
    data = pd.DataFrame(data)
    data.columns = columns
    sensor_dict = load('DataFiles/tagDict_Sensor.pkl')


    means = scaler.center_
    stds = scaler.scale_
    scaler_dict = {columns[i]: {'mean': means[i], 'std': stds[i]} for i in range(len(columns))}
    initial_dict = {'br_7120_ft_1002': 0.8944417746724614, 'bj_7110_ft_1012': 301.5340451940437,
                    'bg_7110_dt_1011_solido': 26.284736257562823, 'bk_7110_ft_1030':114.11711666717109,
                    'bp_7110_ot_1003': 23.43838784136062, 'bo_7110_lt_1009_s4': 4.890675875610863,
                    'bq_7110_pt_1010':96.8623790324402, 'bi_7110_dt_1030_solido':69.24958898228054}

    init_time = 100
    final_time = init_time + 61
    initial_dict_array = {'br_7120_ft_1002': np.array(data['br_7120_ft_1002'].iloc[init_time: final_time]),
                          'bj_7110_ft_1012': np.array(data['bj_7110_ft_1012'].iloc[init_time: final_time]),
                          'bg_7110_dt_1011_solido': np.array(data['bg_7110_dt_1011_solido'].iloc[init_time: final_time]),
                          'bk_7110_ft_1030': np.array(data['bk_7110_ft_1030'].iloc[init_time: final_time]),
                          'bp_7110_ot_1003': np.array(data['bp_7110_ot_1003'].iloc[init_time: final_time]),
                          'bo_7110_lt_1009_s4': np.array(data['bo_7110_lt_1009_s4'].iloc[init_time: final_time]),
                          'bq_7110_pt_1010': np.array(data['bq_7110_pt_1010'].iloc[init_time: final_time]),
                          'bi_7110_dt_1030_solido': np.array(data['bi_7110_dt_1030_solido'].iloc[init_time: final_time])}

    u_dict = {'br_7120_ft_1002': np.array([1.2 for i in range(10)]),
              'bk_7110_ft_1030': np.array([100 for i in range(10)]),
              'bj_7110_ft_1012': np.array([316 for i in range(10)]),
              'bg_7110_dt_1011_solido': np.array([26 for i in range(10)])}


    model = ModeloRecurrente(bed_variables, pressure_variables, torque_variables, solidC_variables, batch_size=16,
                         preds=60, seqlen=60, parameters=parameters, weighted=False, h_init=False, inversed=False,
                         hidden_size=75, folder='Nets/ModeloRecurrente/', xavier_init=True)

    model = ModeloEncDec(bed_variables, pressure_variables, torque_variables, solidC_variables, batch_size=16,
                                                 preds=60, seqlen=60, parameters=parameters, weighted=False, h_init=False, inversed=False,
                                                  hidden_size=75, folder='Nets/ModeloEncDec/', xavier_init=True)


    control_wrapper = ControlWrapper(model, scaler, scaler_dict)


    control_wrapper.set_initial_conditions(initial_dict=initial_dict_array, batch_size=10, scaled_input=False, array=True)
    outs = []
    for i in range(600):
        if i % 100 == 0:
            print(i)
        if i > 200:
            u_dict = {'br_7120_ft_1002': np.array([1.2 for i in range(10)]),
                      'bk_7110_ft_1030': np.array([150 for i in range(10)]),
                      'bj_7110_ft_1012': np.array([316 for i in range(10)]),
                      'bg_7110_dt_1011_solido': np.array([26 for i in range(10)])
                      }

        if i > 400:
            u_dict = {'br_7120_ft_1002': np.array([0.8 for i in range(10)]),
                      'bk_7110_ft_1030': np.array([150 for i in range(10)]),
                      'bj_7110_ft_1012': np.array([316 for i in range(10)]),
                      'bg_7110_dt_1011_solido': np.array([26 for i in range(10)])
                      }

        out = control_wrapper.run(u_dict, scaled_input=False, cascaded=True)
        outs.append(out[0, :, :])

    outs = np.concatenate(outs, axis=0)
    outs_names = ['Bed', 'Pressure', 'Torque', 'SolidC']
    # for i in range(outs.shape[1]):
    #     plt.figure()
    #     plt.title(outs_names[i])
    #     plt.plot(outs[:, i])
    #     plt.grid()
    # plt.show()


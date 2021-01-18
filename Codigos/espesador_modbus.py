import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from OPC_Client import OPC_client
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from ControlModelWrapper import ControlWrapper
from ModeloEncDec.ModeloMultiple import ModeloEncDec
from pyModbusTCP.client import ModbusClient
from pyModbusTCP import utils
from time import sleep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


flag = 0

############################################## Modelos ##################################################################

def save(obj, name):
    with open(name, 'wb') as file:
        pickle.dump(obj, file)


def load(name):
    with open(name, 'rb') as file:
        return pickle.load(file)


with open('DataFiles/parametrosNew.pkl', 'rb') as file:
    parameters = pickle.load(file)

# Inputs Cama
bed_inputs = ['br_7120_ft_1002', 'bk_7110_ft_1030', 'bj_7110_ft_1012', 'bg_7110_dt_1011_solido',
              'bo_7110_lt_1009_s4']
bed_outputs = ['bo_7110_lt_1009_s4']

# Inputs Presión
pressure_inputs = ['br_7120_ft_1002', 'bk_7110_ft_1030', 'bj_7110_ft_1012', 'bg_7110_dt_1011_solido',
                   'bq_7110_pt_1010']
pressure_outputs = ['bq_7110_pt_1010']

# Inputs Presión
torque_inputs = ['br_7120_ft_1002', 'bk_7110_ft_1030', 'bj_7110_ft_1012', 'bg_7110_dt_1011_solido',
                 'bp_7110_ot_1003']
torque_outputs = ['bp_7110_ot_1003']

# Inputs CSólidos
solidC_inputs = ['br_7120_ft_1002', 'bk_7110_ft_1030', 'bq_7110_pt_1010', 'bo_7110_lt_1009_s4',
                 'bp_7110_ot_1003', 'bi_7110_dt_1030_solido']
solidC_outputs = ['bi_7110_dt_1030_solido']

solidC_u = ['bk_7110_ft_1030']
solidC_outputs = ['bi_7110_dt_1030_solido']

bed_variables = [bed_inputs, bed_outputs]
pressure_variables = [pressure_inputs, pressure_outputs]
torque_variables = [torque_inputs, torque_outputs]
solidC_variables = [solidC_inputs, solidC_outputs]



scaler = torch.load('DataFiles/new_data_all.pkl')['scaler']
data = torch.load('DataFiles/new_data_all.pkl')['new_data']
columns = data.columns
data = scaler.inverse_transform(np.array(data))

data = pd.DataFrame(data)
data.columns = columns
sensor_dict = load('DataFiles/tagDict_Sensor.pkl')

scaler.mean_ = scaler.center_
means = scaler.mean_
stds = scaler.scale_
scaler_dict = {columns[i]: {'mean': means[i], 'std': stds[i]} for i in range(len(columns))}


model = ModeloEncDec(bed_variables, pressure_variables, torque_variables, solidC_variables, batch_size=16,
                     preds=60, seqlen=60, parameters=parameters, weighted=False, h_init=False, inversed=False,
                     hidden_size=75, folder='ModeloEncDec/', xavier_init=True)


############################################# Control Wrapper ##########################################################
batch_size = 1
init_time = 100
final_time = init_time + 61
# Condiciones inciales
initial_dict_array = {'br_7120_ft_1002': np.array(data['br_7120_ft_1002'].iloc[init_time: final_time]),
                      'bj_7110_ft_1012': np.array(data['bj_7110_ft_1012'].iloc[init_time: final_time]),
                      'bg_7110_dt_1011_solido': np.array(data['bg_7110_dt_1011_solido'].iloc[init_time: final_time]),
                      'bk_7110_ft_1030': np.array(data['bk_7110_ft_1030'].iloc[init_time: final_time]),
                      'bp_7110_ot_1003': np.array(data['bp_7110_ot_1003'].iloc[init_time: final_time]),
                      'bo_7110_lt_1009_s4': np.array(data['bo_7110_lt_1009_s4'].iloc[init_time: final_time]),
                      'bq_7110_pt_1010': np.array(data['bq_7110_pt_1010'].iloc[init_time: final_time]),
                      'bi_7110_dt_1030_solido': np.array(data['bi_7110_dt_1030_solido'].iloc[init_time: final_time])}

perturbations = np.concatenate([np.array(data['bj_7110_ft_1012'].iloc[final_time:]).reshape(-1, 1),
                                np.array(data['bg_7110_dt_1011_solido'].iloc[final_time:]).reshape(-1, 1)],
                               axis=1)

# Selección del modelo
model_wrapper = ControlWrapper(model, scaler, scaler_dict)

# # Condiciones iniciales del modelo
model_wrapper.set_initial_conditions(initial_dict=initial_dict_array, batch_size=batch_size, scaled_input=False, array=True)

########################################## SIMULACIÓN SENSORES MODBUS TCP ################################################################

class FloatModbusClient(ModbusClient):
    def read_float(self, address, number=1):
        reg_l = self.read_holding_registers(address, number*2)
        if reg_l:
            return[utils.decode_ieee(f)for f in utils.word_list_to_long(reg_l)]
        else:
            return None
    
    def write_float(self, address, floats_list):
        b32_l = [utils.encode_ieee(f) for f in floats_list]
        b16_l = utils.long_list_to_word(b32_l)
        return self.write_multiple_registers(address, b16_l)

cliente = FloatModbusClient(host='localhost', port=12345, auto_open=True) #Esto habría que cambiarlo por la dirreción IP del PC2
cliente_inputs = FloatModbusClient(host='localhost', port=12346, auto_open=True) #Esto habría que cambiarlo por la dirreción IP del PC2


########################################## Simulación ##################################################################
outputs_history = []
inputs_history = []
set_points_list = []

# Condiciones iniciales
cliente_inputs.write_float(0, [0.7,120])

i = 0
index = 0

while True:
    sleep(1)

    # Reading Inputs
    flocculant = cliente_inputs.read_float(0,1)
    output_flow = cliente_inputs.read_float(2,1)
    print(flocculant)

    # Reading perturbations from real data
    input_flow = perturbations[index, 0]
    input_solidC = perturbations[index, 1]

    # Writing perturbations to MODBUS
    cliente.write_float(0, [flocculant,output_flow,input_flow,input_solidC])


    u_dict = {'br_7120_ft_1002': np.array([flocculant]), 'bk_7110_ft_1030': np.array([output_flow]),
              'bj_7110_ft_1012': np.array([input_flow]), 'bg_7110_dt_1011_solido': np.array([input_solidC])}

    out = model_wrapper.run(u_dict, scaled_input=False, cascaded=True, scaled_output=False)

    # Introducing sensor noise
    output = out + np.concatenate([np.random.normal(loc=0, scale=0.2, size=(1, 1, 1)), np.random.normal(loc=0, scale=0.2, size=(1, 1, 3))], axis=2)
    output = output.squeeze()
    print(output)
    outputs_history.append(output.reshape(1, -1))

    # Se escriben datos en servidor modbus
    cliente.write_float(8, [output[0],output[1],output[2],output[3]])

    index += 1

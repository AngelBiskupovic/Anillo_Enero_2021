from pyModbusTCP.server import ModbusServer, DataBank
from time import sleep
from random import uniform
from pyModbusTCP import utils
import statistics as stats
import random
import time

try:
    from IPython import embed
except ImportError:
    import code


    def embed():
        vars = globals()
        vars.update(locals())
        shell = code.InteractiveConsole(vars)
        shell.interact()

from opcua import Client


##############Funcion de lectura y escritura modbus###################
def read_float(address, number=1):
    reg_l = DataBank.get_words(address, number*2)
    if reg_l:
        return[utils.decode_ieee(f)for f in utils.word_list_to_long(reg_l)]
    else:
        return None
    
def write_float(address, floats_list):
    b32_l = [utils.encode_ieee(f) for f in floats_list]
    b16_l = utils.long_list_to_word(b32_l)
    return DataBank.set_words(address, b16_l)

##############Servidor Modbus###################
servidor = ModbusServer(host="localhost", port=12345, no_block=True) ##Esto habría que cambiarlo por la dirección IP del PC 2

servidor.start()
##############Cliente OPC###################
url = "opc.tcp://192.168.0.15:4080"
client = Client(url)
client.connect()

client.conect()
flag = 1

inputs = {'flocculant': 0, 'output_flow': 0}
outputs = {'solidC': 0, 'bed': 0, 'pressure': 0, 'torque': 0}
perturbations = {'input_flow': 0, 'solid_concentration': 0}

root = client.get_root_node()
objects = client.get_objects_node()
thickener = objects.get_child(['2:Thickener'])


inputs_folder = thickener.get_child(['2:Inputs'])
inputs['flocculant'] = inputs_folder.get_child(['2:flocculant'])
inputs['output_flow'] = inputs_folder.get_child(['2:output_flow'])

outputs_folder = thickener.get_child(['2:Outputs'])
outputs['solidC'] = outputs_folder.get_child('2:solidC')
outputs['bed'] = outputs_folder.get_child('2:bed')
outputs['pressure'] = outputs_folder.get_child('2:pressure')
outputs['torque'] = outputs_folder.get_child('2:torque')

perturbations_folder = thickener.get_child(['2:Perturbations'])
perturbations['input_flow'] = perturbations_folder.get_child(['2:input_flow'])
perturbations['solid_concentration'] = perturbations_folder.get_child(['2:solid_concentration'])

####Creacion de lista de variables para reducción de tiempo de envio de 1 a 5 sseg##########
flocculant1_list = []
output_flow1_list = []
input_flow_list = []
input_solidC_list = []
bed_list = []
pressure_list = []
torque_list = []
solidC_list = []

###############################Proceso de envio y recepción de datos######################################

while True:
    
    sleep(1)
    #Lectura variables desde cliente modbus
    Variables = read_float(0,8)
    print(Variables)
    
    #inputs
    flocculant1 = Variables[0]
    output_flow1 = Variables[1]

    #Perturbaciones
    input_flow = Variables[2]
    input_solidC = Variables[3]

    #Salidas
    bed = Variables[4]
    pressure = Variables[5]
    torque = Variables[6]
    solidC = Variables[7]

    if len(flocculant1_list) < 5:
        
        if input_solidC !=0:

            flocculant1_list.append(flocculant1)
            output_flow1_list.append(output_flow1)
            input_flow_list.append(input_flow)
            input_solidC_list.append(input_solidC)
            bed_list.append(bed)
            pressure_list.append(pressure)
            torque_list.append(torque)
            solidC_list.append(solidC)
        
        else:
            flocculant1_list = flocculant1_list
            output_flow1_list = output_flow1_list
            input_flow_list = input_flow_list
            input_solidC_list = input_solidC_list
            bed_list = bed_list
            pressure_list = pressure_list
            torque_list = torque_list
            solidC_list = solidC_list         

    else:
        flocculant1_mean = stats.mean(flocculant1_list)
        flocculant1_list = [flocculant1]    
        output_flow1_mean = stats.mean(output_flow1_list)
        output_flow1_list = [output_flow1] 
        input_flow_mean = stats.mean(input_flow_list)
        input_flow_list = [input_flow] 
        input_solidC_mean = stats.mean(input_solidC_list)
        input_solidC_list = [input_solidC] 
        bed_mean = stats.mean(bed_list)
        bed_list = [bed] 
        pressure_mean = stats.mean(pressure_list)
        pressure_list = [pressure] 
        torque_mean = stats.mean(torque_list)
        torque_list = [torque] 
        solidC_mean = stats.mean(solidC_list)
        solidC_list = [solidC]  
        ##########Envio MODBUS-OPC##########


        #Entradas a OPC
        inputs['flocculant'].set_value(float(flocculant1_mean))
        inputs['output_flow'].set_value(float(output_flow1_mean))
        #Perturbarciones a OPC
        perturbations['input_flow'].set_value(float(input_flow_mean))
        client.perturbations['solid_concentration'].set_value(float(input_solidC_mean))
        #Salidas a OPC
        outputs['bed'].set_value(float(bed_mean))
        outputs['pressure'].set_value(float(pressure_mean))
        outputs['torque'].set_value(float(torque_mean))
        outputs['solidC'].set_value(float(solidC_mean))
        
        ##########Envio OPC-MODBUS##########
        #Recepcion desde cliente OPC
        flocculant = inputs['flocculant'].get_value()
        output_flow = inputs['output_flow'].get_value()

        #envio a cliente modbus
        write_float(0,[flocculant,output_flow])
    
servidor.close()

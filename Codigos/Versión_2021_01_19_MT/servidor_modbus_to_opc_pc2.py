from pyModbusTCP.server import ModbusServer, DataBank
from time import sleep
from random import uniform
from pyModbusTCP import utils
from OPC_Client import OPC_client
import statistics as stats



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
servidor = ModbusServer(host="localhost", port=12346, no_block=True)
#servidor_inputs = ModbusServer(host="localhost", port=12346, no_block=True)

servidor.start()
#servidor_inputs.start()
##############Cliente OPC###################
client = OPC_client("opc.tcp://localhost:4080")

client.conect()
flag =1 
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

        # intentar conectar al servidor UA
        try:
            if flag == 0:
                client.connect()
                flag = 1

            ##########Envio MODBUS-OPC##########
            #Entradas a OPC
            client.inputs['flocculant'].set_value(float(flocculant1_mean) + uniform(0.000001, 0.00011))
            client.inputs['output_flow'].set_value(float(output_flow1_mean) +uniform(0.000001, 0.00011))
            #Perturbarciones a OPC
            client.perturbations['input_flow'].set_value(float(input_flow_mean))
            client.perturbations['solid_concentration'].set_value(float(input_solidC_mean))
            # #Salidas a OPC
            client.outputs['bed'].set_value(float(bed_mean))
            client.outputs['pressure'].set_value(float(pressure_mean))
            client.outputs['torque'].set_value(float(torque_mean))
            client.outputs['solidC'].set_value(float(solidC_mean))
            
            ##########Envio OPC-MODBUS##########
            #Recepcion desde cliente OPC
            flocculant = client.inputs['flocculant'].get_value()
            output_flow = client.inputs['output_flow'].get_value()

            #envio a cliente modbus
            write_float(0,[flocculant,output_flow])    


        # De fallar la conexión opcua, intentar nuevamente
        except Exception as e:
            print(e)
            flag = 0
            sleep(2)

    
servidor.close()
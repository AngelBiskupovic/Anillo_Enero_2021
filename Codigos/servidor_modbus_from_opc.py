from pyModbusTCP.server import ModbusServer, DataBank
from time import sleep
from random import uniform
from pyModbusTCP import utils
from OPC_Client import OPC_client
import statistics as stats

class SubHandler(object):

    """
    Subscription Handler. To receive events from server for a subscription
    """
    def __init__(self):
       pass

    def datachange_notification(self, node, val, data):
        global flag
        flag = 1

    def event_notification(self, event):
        print("Python: New event", event)

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
servidor = ModbusServer(host="localhost", port=12345, no_block=True)
servidor_inputs = ModbusServer(host="localhost", port=12346, no_block=True)

servidor.start()
servidor_inputs.start()
##############Cliente OPC###################
client = OPC_client("opc.tcp://localhost:4840/freeopcua/server/", subscribe_to='inputs', handler=SubHandler)

client.conect()

###############################Proceso de envio y recepción de datos######################################

while True:
    sleep(5)        
    ##########Envio OPC-MODBUS##########
    #Recepcion desde cliente OPC
    flocculant = client.inputs['flocculant'].get_value()
    output_flow = client.inputs['output_flow'].get_value()

    #envio a cliente modbus
    write_float(0,[flocculant,output_flow])
    
servidor.close()
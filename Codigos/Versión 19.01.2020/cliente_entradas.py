from pyModbusTCP.client import ModbusClient
from pyModbusTCP import utils
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

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

cliente = FloatModbusClient(host='localhost', port=12345, auto_open=True)
cliente_to_server = FloatModbusClient(host='localhost', port=12346, auto_open=True)

while True:
    sleep(1)
    Variables = cliente.read_float(0,2)
    flocculant = Variables[0]
    output_flow = Variables[1]
    cliente_to_server.write_float(0,[flocculant,output_flow])
    print(Variables)
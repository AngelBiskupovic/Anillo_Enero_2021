from pyModbusTCP.client import ModbusClient
from pyModbusTCP import utils
import sys
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
    Variables = cliente.read_float(8,4)
    bed = Variables[0]
    pressure = Variables[1]
    torque = Variables[2]
    solidC = Variables[3]
    cliente_to_server.write_float(8,[bed,pressure,torque,solidC])
    print(Variables)
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")
import logging
from datetime import datetime
import time
import numpy as np
from cassandra.cluster import Cluster
from cassandra import timestamps
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

# Conexión a Base de Datos Cassandra

cluster = Cluster(['192.168.50.101'], port=9842)
keyspace = 'thickener'
connection = cluster.connect(keyspace)



def cassandra(val, node_name):

    datetime_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S+00")

    if node_name == 'flocculant':
        name = 'flocculant'
    if node_name == 'output_flow':
        name = 'output_flow'
    if node_name == 'input_flow':
        name = 'input_flow'
    if node_name == 'solid_concentration':
        name = 'solid_concentration'
    if node_name == 'solidC':
        name = 'solidC'
    if node_name == 'bed':
        name = 'bed'
    if node_name == 'pressure':
        name = 'pressure'
    if node_name == 'torque':
        name = 'torque'


    connection.execute("""
     INSERT INTO thickener.modbus_sensors (sensor, datetime, value)
     VALUES (%(sensor)s, %(datetime)s, %(value)s)
     """,
     {'sensor': str(name), 'datetime':datetime_now, 'value': val}
    )


class SubHandler(object):

    # Función que adquiere el nodo actualizado
    # por el cliente OPC UA

    def datachange_notification(self, node, val, data):


        # if node == flocculant:
        #rint(node, val)

        if node == flocculant:
            print('inserting value {} from sensor: {} '.format(val,flocculant_name))
            cassandra(val, flocculant_name)

        elif node == output_flow:
            print('inserting value {} from sensor: {} '.format(val,output_flow_name))
            cassandra(val, output_flow_name)

        elif node == input_flow:
            print('inserting value {} from sensor: {} '.format(val, input_flow_name))
            cassandra(val, input_flow_name)

        elif node == solid_concentration:
            print('inserting value {} from sensor: {} '.format(val, solid_concentration_name))        
            cassandra(val, solid_concentration_name)

        elif node == solidC:
            print('inserting value {} from sensor: {} '.format(val,solidC_name))
            cassandra(val, solidC_name)

        elif node == bed:
            print('inserting value {} from sensor: {} '.format(val, bed_name))
            cassandra(val, bed_name)
        elif node == pressure:
            print('inserting value {} from sensor: {} '.format(val, pressure_name))
            cassandra(val, pressure_name)
        elif node == torque:
            print('inserting value {} from sensor: {} '.format(val, torque_name))
            cassandra(val, torque_name)


    def event_notification(self, event):
        print("Python: New event", event)



if __name__ == "__main__":

    url = "opc.tcp://0.0.0.0:4080"

    inputs = {'flocculant': 0, 'output_flow': 0}
    perturbations = {'input_flow': 0, 'solid_concentration': 0}
    outputs = {'solidC': 0, 'bed': 0, 'pressure': 0, 'torque': 0}


    print("conectando")
    client = Client(url)
    client.connect()

    root = client.get_root_node()
    objects = client.get_objects_node()
    thickener = objects.get_child(['2:Thickener'])


    inputs_folder = thickener.get_child(['2:Inputs'])
    perturbations_folder = thickener.get_child(['2:Perturbations'])
    outputs_folder = thickener.get_child(['2:Outputs'])


    inputs['flocculant'] = inputs_folder.get_child(['2:flocculant'])
    inputs['output_flow'] = inputs_folder.get_child(['2:output_flow'])


    flocculant = inputs['flocculant'] 
    output_flow = inputs['output_flow']

    flocculant_name = flocculant.get_display_name().Text
    output_flow_name = output_flow.get_display_name().Text



    perturbations['input_flow'] = perturbations_folder.get_child(['2:input_flow'])
    perturbations['solid_concentration'] = perturbations_folder.get_child(['2:solid_concentration'])


    input_flow = perturbations['input_flow'] 
    solid_concentration = perturbations['solid_concentration']

    input_flow_name = input_flow.get_display_name().Text
    solid_concentration_name = solid_concentration.get_display_name().Text



    outputs['solidC'] = outputs_folder.get_child(['2:solidC'])
    outputs['bed'] = outputs_folder.get_child(['2:bed'])
    outputs['pressure'] = outputs_folder.get_child(['2:pressure'])
    outputs['torque'] = outputs_folder.get_child(['2:torque'])

    solidC = outputs['solidC'] 
    bed = outputs['bed']
    pressure = outputs['pressure']
    torque = outputs['torque']

    solidC_name = solidC.get_display_name().Text
    bed_name = bed.get_display_name().Text
    pressure_name = pressure.get_display_name().Text
    torque_name = torque.get_display_name().Text    

    try:

        handler_inputs = SubHandler()
        hanfler_perturbations = SubHandler() 
        handler_outputs = SubHandler()

        sub_inputs = client.create_subscription(500, handler_inputs)
        sub_perturbations = client.create_subscription(500, hanfler_perturbations)
        sub_outputs = client.create_subscription(500, handler_outputs)


        for key, var in inputs.items():
            handle = sub_inputs.subscribe_data_change(var)
        for key, var in perturbations.items():
            handle = sub_perturbations.subscribe_data_change(var) 
        for key, var in outputs.items():
            handle = sub_outputs.subscribe_data_change(var)                       

        # sub.unsubscribe(handle)
        # sub.delete()
        embed()

    finally:
        client.disconnect()

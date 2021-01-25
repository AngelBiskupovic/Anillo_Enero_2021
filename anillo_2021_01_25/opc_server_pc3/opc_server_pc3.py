# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")
import logging
from datetime import datetime
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

from opcua import ua, uamethod, Server

# Conexión a Base de Datos Cassandra



class SubHandler(object):

    # Función que adquiere el nodo actualizado
    # por el cliente OPC UA

    def datachange_notification(self, node, val, data):

        #flocculant.get_browse_name()
        if node == flocculant:
            print('new value {} from node: {} '.format(val,flocculant.get_display_name().Text))
        elif node == output_flow:
            print('new value {} from node: {} '.format(val,output_flow.get_display_name().Text))
        elif node == solidC:
            print('new value {} from node: {} '.format(val,solidC.get_display_name().Text))
        elif node == bed:
            print('new value {} from node: {} '.format(val, bed.get_display_name().Text))
        elif node == pressure:
            print('new value {} from node: {} '.format(val, pressure.get_display_name().Text))
        elif node == torque:
            print('new value {} from node: {} '.format(val, torque.get_display_name().Text))
        elif node == input_flow:
            print('new value {} from node: {} '.format(val, input_flow.get_display_name().Text))
        elif node == solid_concentration:
            print('new value {} from node: {} '.format(val, solid_concentration.get_display_name().Text))

    def event_notification(self, event):
        print("Python: New event", event)



if __name__ == "__main__":
    server = Server()

    #url = "opc.tcp://192.168.0.3:4080"
    url = "opc.tcp://localhost:4080"

    server.set_endpoint(url)
    name = "Thickener"
    addspace = server.register_namespace(name)
    objects = server.get_objects_node()
    thickener = objects.add_folder(addspace, 'Thickener')

    inputs = thickener.add_object(addspace, 'Inputs')
    perturbations = thickener.add_object(addspace, 'Perturbations')
    outputs = thickener.add_object(addspace, 'Outputs')

    input_flow = perturbations.add_variable(addspace, 'input_flow', 0)
    solid_concentration = perturbations.add_variable(addspace, 'solid_concentration', 0)

    input_flow.set_writable()
    solid_concentration.set_writable()

    #flocculant = inputs.add_variable(addspace, 'flocculant', np.repeat(float('nan'), 2).tolist())
    flocculant = inputs.add_variable(addspace, 'flocculant', 0)
    output_flow = inputs.add_variable(addspace, 'output_flow', 0)

    flocculant.set_writable()
    output_flow.set_writable()


    solidC = outputs.add_variable(addspace, 'solidC', 0)
    bed = outputs.add_variable(addspace, 'bed', 0)
    pressure = outputs.add_variable(addspace, 'pressure', 0)
    torque = outputs.add_variable(addspace, 'torque', 0)

    solidC.set_writable()
    bed.set_writable()
    pressure.set_writable()
    torque.set_writable()

    perturbations = [input_flow, solid_concentration]
    inputs = [flocculant, output_flow]
    outputs = [solidC, bed, pressure, torque]
    # Se inicia el servidor
    server.start()

    # Prueba de comunicación
    # de fallar se detiene el servidor
    try:

        # Se habilita el servidor a suscribirse
        handler_inputs = SubHandler()
        handler_outputs = SubHandler()
        handler_perturbations = SubHandler()


        sub_inputs = server.create_subscription(100, handler_inputs)
        sub_outputs = server.create_subscription(100, handler_outputs)
        sub_perturbations = server.create_subscription(100, handler_perturbations)

        for i in range(len(inputs)):
            handle = sub_inputs.subscribe_data_change(inputs[i])

        for i in range(len(outputs)):
            handle = sub_outputs.subscribe_data_change(outputs[i])

        for i in range(len(perturbations)):
            handle = sub_perturbations.subscribe_data_change(perturbations[i])

        embed()
    finally:
        server.stop()

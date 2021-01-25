from opcua import ua, Client
import time

# class SubHandler(object):

#     """
#     Subscription Handler. To receive events from server for a subscription
#     """
#     def __init__(self):
#        pass

#     def datachange_notification(self, node, val, data):
#         print("Python: New data change event", node, val)

#     def event_notification(self, event):
#         print("Python: New event", event)


class OPC_client:
    def __init__(self, address):
        self.address = address
        self.client = Client(self.address, timeout=15)
        self.inputs = {'flocculant': 0, 'output_flow': 0}
        self.perturbations = {'input_flow': 0, 'solid_concentration': 0}
        self.outputs = {'solidC': 0, 'bed': 0, 'pressure': 0, 'torque': 0}

    def instantiate(self):
        self.root = self.client.get_root_node()
        self.objects = self.client.get_objects_node()
        self.thickener = self.objects.get_child(['2:Thickener'])
        self.inputs_folder = self.thickener.get_child(['2:Inputs'])
        self.perturbations_folder = self.thickener.get_child(['2:Perturbations'])
        self.outputs_folder = self.thickener.get_child(['2:Outputs'])

        self.inputs['flocculant'] = self.inputs_folder.get_child(['2:flocculant'])
        self.inputs['output_flow'] = self.inputs_folder.get_child(['2:output_flow'])

        self.perturbations['input_flow'] = self.perturbations_folder.get_child(['2:input_flow'])
        self.perturbations['solid_concentration'] = self.perturbations_folder.get_child(['2:solid_concentration'])

        self.outputs['solidC'] = self.outputs_folder.get_child(['2:solidC'])
        self.outputs['bed'] = self.outputs_folder.get_child(['2:bed'])
        self.outputs['pressure'] = self.outputs_folder.get_child(['2:pressure'])
        self.outputs['torque'] = self.outputs_folder.get_child(['2:torque'])


    def conect(self):
        try:
            self.client.connect()
            self.objects = self.client.get_objects_node()
            print('OPC client has been connected to the server')
            self.instantiate()
        except:
            self.client.disconnect()
            print('OPC client hasn t be able to connect to the server')


if __name__ == '__main__':
    client = OPC_client("opc.tcp://localhost:4080")
    client.conect()

    valor = client.inputs['flocculant'].get_value()
    print('valor: {}'.format(valor))

    client.outputs['solidC'].set_value(120)
    client.inputs['flocculant'].set_value(0.7)









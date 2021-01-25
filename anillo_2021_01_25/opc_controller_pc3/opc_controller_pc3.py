# -*- coding: utf-8 -*-

## CONECTOR DCS-OPCUA

import random
import time
from datetime import datetime
from cassandra.cluster import Cluster
from cassandra import timestamps
import pandas as pd
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

def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)
# Conexión a Base de Datos Cassandra

cluster = Cluster(['192.168.0.15'])
keyspace = 'thickener'
connection = cluster.connect(keyspace)
connection.row_factory = pandas_factory
connection.default_fetch_size = None

inputs_tags = ['flocculant', 'output_flow']
df_tot = [0] *len(inputs_tags)


def query():

    df_tot = [0] * len(inputs_tags)
    for r in range(0, len(inputs_tags)):
        rows = connection.execute("select * from modbus_sensors where sensor = '{}' order by datetime desc LIMIT 1".format(inputs_tags[r]))
        df = rows._current_rows
        df = df.iloc[::-1]
        df = df.reset_index(drop=True)
        df_tot[r] = df

    df_tot = pd.concat(df_tot, ignore_index=True)
    df_tot = pd.pivot_table(df_tot, values='value', columns='sensor', index='datetime')
    df_tot = df_tot.reset_index(drop=False)
    sp1=df_tot.loc[0,['flocculant']].values[0]
    sp2=df_tot.loc[0,['output_flow']].values[0]


    return sp1, sp2

class SubHandler(object):
    """
    Subscription Handler. To receive events from server for a subscription
    data_change and event methods are called directly from receiving thread.
    Do not do expensive, slow or network operation there. Create another
    thread if you need to do such a thing
    """



if __name__ == "__main__":

    #Habilitar avisos de alertas

    #url = "opc.tcp://192.168.0.3:4080"
    url = "opc.tcp://localhost:4080"

    inputs = {'flocculant': 0, 'output_flow': 0}


    print("conectando")
    client = Client(url)
    client.connect()
    flag = 1
    # print("Client Connected")



    #Repetir infinitamente
    while True:

        # intentar conectar al servidor UA
        try:
            if flag == 0:
                client.connect()
                flag = 1

            #flocculant = [time.mktime(datetime.now().timetuple()) * 1000, 1 + random.uniform(0.000001, 0.00011)]

            #[sp1,sp2]=query()

            #flocculant = [float(sp1) + 1  ]
            #output_flow = [float(sp2) + 10 ]
            root = client.get_root_node()
            objects = client.get_objects_node()
            thickener = objects.get_child(['2:Thickener'])


            inputs_folder = thickener.get_child(['2:Inputs'])
            inputs['flocculant'] = inputs_folder.get_child(['2:flocculant'])
            inputs['output_flow'] = inputs_folder.get_child(['2:output_flow'])

            flocculant = 1 +random.uniform(0.5,2.5)
            output_flow = 2 +random.uniform(50,150)

            inputs['flocculant'].set_value(flocculant)
            inputs['output_flow'].set_value(output_flow)


            print(flocculant, output_flow)
            time.sleep(10)

        except Exception as e:
            print(e)
            flag = 0
            time.sleep(2)


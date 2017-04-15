from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import time
import subprocess
import numpy as np
import os
import sys
import tensorflow as tf
import cifar10
import cifar10_input
import asyncio

FLAGS = tf.app.flags.FLAGS

class MasterProcess(asyncio.Protocol):
    def __init__(self):
        self.transports = []

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        print("Commander connected to {}".format(peername))
        self.transports.append(transport)

    def data_received(self, data):
        print("Received:", data.decode())

    def connection_lost(self, exc):
        pass

class WorkerProcess(asyncio.Protocol):
    def __init__(self):
        pass

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        print("Worker connected to {}".format(peername))
        self.transport = transport

    def data_received(self, data):
        pass

    def connection_lost(self, exc):
        pass

def master_start_server(master_host, tf_session, tf_ops):
    print("Master starting UDP server")
    loop = asyncio.get_event_loop()
    ip_addr = master_host.split(":")[0]
    port = int(master_host.split(":")[1])
    master = MasterProcess()
    listen_commander = loop.create_server(
        lambda : master,
        ip_addr, port)
    server = loop.run_until_complete(listen_commander)
    return master, loop

def worker_start_server(master_host, tf_session, tf_ops):
    print("Worker starting UDP client")
    loop = asyncio.get_event_loop()
    ip_addr = master_host.split(":")[0]
    port = int(master_host.split(":")[1])
    listen_command_receiver = loop.create_connection(lambda: WorkerProcess(),
                                                     ip_addr, port)
    _, worker_process = loop.run_until_complete(listen_command_receiver)
    return worker_process, loop

def train():

    # Basic distributed training flags
    assert(FLAGS.hosts != '')
    assert(FLAGS.machine_index != -1)
    index = FLAGS.machine_index
    hosts = FLAGS.hosts.split(",")
    is_master = FLAGS.machine_index == 0

    # Load data set
    """images_train_raw, labels_train_raw, images_test_raw, labels_test_raw = cifar10_input.load_cifar_data_raw()
    random_permutation = np.random.permutation(images_train_raw.shape[0])
    images_train_raw = images_train_raw[random_permutation]
    labels_train_raw = labels_train_raw[random_permutation]"""

    # Basic model creation for cuda convnet
    scope_name = "parameters_1"
    with tf.variable_scope(scope_name):
        images = tf.placeholder(tf.float32, shape=(None, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, cifar10.NUM_CHANNELS))
        labels = tf.placeholder(tf.int32, shape=(None,))
        logits = cifar10.inference(images)
        loss_op = cifar10.loss(logits, labels, scope_name)
        train_op = cifar10.train(loss_op, scope_name)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # UDP connection
    if is_master:
        master_process, loop = master_start_server(hosts[0], None, None)
    else:
        while True:
            try:
                worker_process, loop  = worker_start_server(hosts[0], None, None)
                break
            except:
                time.sleep(1)

    loop.run_forever()

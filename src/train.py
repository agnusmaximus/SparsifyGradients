from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import time
import socket
import binascii
import subprocess
import numpy as np
import os
import sys
import tensorflow as tf
import cifar10
import cifar10_input
import pickle
from mpi4py import MPI

FLAGS = tf.app.flags.FLAGS

def synchronize_model(sess, variables, com, rank, assignment_op, placeholders):
    materialized_variables = []
    if rank == 0:
        print("Master materializing variables...")

        # Materialize variables
        for variable in variables:
            materialized_variables.append(sess.run([variable])[0])

    materialized_variables = com.bcast(materialized_variables, root=0)

    # Update variables
    if rank != 0:
        print("Worker setting variables")
        assert(len(materialized_variables) == len(placeholders))
        feed_dict = {placeholders[i] : materialized_variables[i] for i in range(len(placeholders))}
        sess.run(assignment_op, feed_dict=feed_dict)

def get_next_batch(fractional_images, fractional_labels, cur_index, batch_size):
  start = cur_index
  end = min(cur_index+batch_size, fractional_labels.shape[0])
  next_index = end
  next_batch_images = fractional_images[start:end]
  next_batch_labels = fractional_labels[start:end]

  # Wrap around
  wraparound_images = np.array([])
  wraparound_labels = np.array([])
  if end-start < batch_size:
    next_index = batch_size-(end-start)
    wraparound_images = fractional_images[:next_index]
    wraparound_labels = fractional_labels[:next_index]

  assert(wraparound_images.shape[0] == wraparound_labels.shape[0])
  if wraparound_images.shape[0] != 0:
    next_batch_images = np.vstack((next_batch_images, wraparound_images))
    next_batch_labels = np.hstack((next_batch_labels, wraparound_labels))

  assert(next_batch_images.shape[0] == batch_size)
  assert(next_batch_labels.shape[0] == batch_size)

  return next_batch_images, next_batch_labels, next_index % fractional_labels.shape[0]

# Helper function to load feed dictionary
def get_feed_dict(batch_size, images_raw, labels_raw, images, labels):
    images_real, labels_real, next_index = get_next_batch(images_raw, labels_raw,
                                                         get_feed_dict.fractional_dataset_index,
                                                         batch_size)
    get_feed_dict.fractional_dataset_index = next_index
    assert(images_real.shape[0] == batch_size)
    assert(labels_real.shape[0] == batch_size)
    return {images : images_real, labels: labels_real}

def train():

    # Communication defines
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    addr = socket.gethostbyname(socket.gethostname())

    print("Worker %d with address %s" % (rank, str(addr)))

    # Load data set
    images_train_raw, labels_train_raw, images_test_raw, labels_test_raw = cifar10_input.load_cifar_data_raw(rank)
    if rank != 0:
        random_permutation = np.random.permutation(images_train_raw.shape[0])
        images_train_raw = images_train_raw[random_permutation]
        labels_train_raw = labels_train_raw[random_permutation]

    # Basic model creation for cuda convnet
    scope_name = "parameters_1"
    with tf.variable_scope(scope_name):
        images = tf.placeholder(tf.float32, shape=(None, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, cifar10.NUM_CHANNELS))
        labels = tf.placeholder(tf.int32, shape=(None,))
        logits = cifar10.inference(images)
        loss_op = cifar10.loss(logits, labels, scope_name)
        train_op, grads_op = cifar10.train(loss_op, scope_name)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

    model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
    model_variables_placeholders = [tf.placeholder(dtype=x.dtype, shape=x.get_shape()) for x in model_variables]
    model_variables_assign = [tf.assign(model_variables[i], model_variables_placeholders[i]) for i in range(len(model_variables))]

    with tf.Session() as sess:

        tf.initialize_all_variables().run()
        tf.train.start_queue_runners(sess=sess)

        get_feed_dict.fractional_dataset_index = 0

        while True:
            # Synchronize model
            synchronize_model(sess, model_variables, comm, rank, model_variables_assign, model_variables_placeholders)

            # Perform distributed gradient descent
            if rank != 0:
                fd = get_feed_dict(FLAGS.batch_size, images_train_raw, labels_train_raw, images, labels)
                materialized_gradients = sess.run([grads_op], feed_dict=fd)

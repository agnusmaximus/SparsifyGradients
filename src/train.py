from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
import socket
import binascii
import subprocess
import numpy as np
import os
import sys
import tensorflow as tf
import cifar10
import cifar10_input
import io
import pickle
from scipy import sparse
from mpi4py import MPI

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('sparsify', True,
                         """To sparsify gradients""")
tf.app.flags.DEFINE_bool('save_gradient_magnitude_histogram', False,
                         """Save gradient magnitude histogram""")
tf.app.flags.DEFINE_bool('variable_cutoff', True,
                         """To sparsify gradients""")
tf.app.flags.DEFINE_integer('cutoff', 90,
                            """To sparsify gradients""")
tf.app.flags.DEFINE_integer('n_iterations', 10000000,
                            """Num iterations""")
tf.app.flags.DEFINE_integer('n_epochs', 100000,
                            """Num iterations""")
tf.app.flags.DEFINE_float('accuracy_to_reach', .995, "Accuracy to reach")

def get_variable_cutoff(vals, percent_acc):
    if percent_acc <= .80:
        return 90
    return 80

def get_variable_sparsified_grads(materialized_grads, percent_acc):
    thresholds = [np.percentile(abs(x), get_variable_cutoff(abs(x), percent_acc)) for x in materialized_grads]
    sparsified = [x * (abs(x) > threshold) for x, threshold in zip(materialized_grads, thresholds)]
    sparsified_flatten = [x.flatten() for x in sparsified]
    return [sparse.csr_matrix(x) for x in sparsified_flatten]

def aggregate_and_apply_gradients(sess, variables, com, rank, n_workers, materialized_grads, apply_gradients_placeholder, apply_gradients_op, percent_acc):
    if FLAGS.sparsify and rank != 0:
        if FLAGS.variable_cutoff:
            materialized_grads = get_variable_sparsified_grads(materialized_grads, percent_acc)
        else:
            thresholds = [np.percentile(abs(x), FLAGS.cutoff) for x in materialized_grads]
            sparsified = [x * (abs(x) > threshold) for x, threshold in zip(materialized_grads, thresholds)]
            sparsified_flatten = [x.flatten() for x in sparsified]
            materialized_grads = [sparse.csr_matrix(x) for x in sparsified_flatten]

    all_gradients = com.gather(materialized_grads, root=0)
    if rank == 0:
        for worker in range(1, n_workers):
            if FLAGS.sparsify:
                # Decode sparse matrix
                worker_gradients = [np.reshape(np.asarray(x.todense()), variables[i].get_shape().as_list()) for i, x in enumerate(all_gradients[worker])]
            else:
                worker_gradients = all_gradients[worker]
            fd = {apply_gradients_placeholder[i] : worker_gradients[i] for i in range(len(apply_gradients_placeholder))}
            sess.run(apply_gradients_op, feed_dict=fd)

def synchronize_model(sess, variables, com, rank, assignment_op, placeholders):
    materialized_variables = []
    if rank == 0:

        # Materialize variables
        for variable in variables:
            materialized_variables.append(sess.run([variable])[0])

    materialized_variables = com.bcast(materialized_variables, root=0)

    # Update variables
    if rank != 0:
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

    try:
        addr = socket.gethostbyname(socket.gethostname())
        print("Worker %d with address %s" % (rank, str(addr)))
    except:
        pass

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
        train_op, grads_and_vars, opt = cifar10.train(loss_op, scope_name)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

    model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
    model_variables_placeholders = [tf.placeholder(dtype=x.dtype, shape=x.get_shape()) for x in model_variables]
    model_variables_assign = [tf.assign(model_variables[i], model_variables_placeholders[i]) for i in range(len(model_variables))]

    apply_gradients_placeholders = [tf.placeholder(dtype=grad.dtype, shape=grad.get_shape()) for grad, var in grads_and_vars]
    apply_gradients_op = opt.apply_gradients(zip(apply_gradients_placeholders, [var for grad, var in grads_and_vars]))

    with tf.Session() as sess:

        tf.initialize_all_variables().run()
        tf.train.start_queue_runners(sess=sess)

        get_feed_dict.fractional_dataset_index = 0
        n_examples_processed = 0
        iteration = 0
        eval_iteration_interval = int(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / (FLAGS.batch_size * (size-1)))
        evaluate_times = []
        t_start = time.time()

        sync_variables_times = 0
        accumulate_gradients_times = 0
        compute_times = 0
        previous_accuracy = 0

        for i in range(FLAGS.n_iterations):

            cur_epoch = n_examples_processed / cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

            if cur_epoch >= FLAGS.n_epochs:
                break

            # Synchronize model
            t_synchronize_start = time.time()
            synchronize_model(sess, model_variables, comm, rank, model_variables_assign, model_variables_placeholders)
            t_synchronize_end = time.time()
            sync_variables_times += t_synchronize_end-t_synchronize_start

            if rank == 0 and iteration % 100 == 0:
                print("Epoch: %f" % (cur_epoch))

            if rank == 0:
                mean_sync = sync_variables_times / (iteration+1)
                mean_compute = compute_times / (iteration+1)
                mean_acc_gradients = accumulate_gradients_times / (iteration+1)
                print("Mean sync time: %f" % mean_sync)
                print("Mean compute time: %f" % mean_compute)
                print("Mean acc gradients time: %f" % mean_acc_gradients)

            if iteration % eval_iteration_interval == 0:

                # Evaluate on master
                if rank == 0 and iteration != 0:
                    print("Master evaluating...")
                    acc_total, loss_total = 0, 0
                    evaluate_t_start = time.time()
                    for i in range(0, cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, FLAGS.evaluate_batchsize):
                        print("%d of %d" % (i, cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN))
                        fd = get_feed_dict(FLAGS.evaluate_batchsize, images_train_raw, labels_train_raw, images, labels)
                        acc_p, loss_p = sess.run([top_k_op, loss_op], feed_dict=fd)
                        acc_total += np.sum(acc_p)
                        loss_total += loss_p
                    evaluate_t_end = time.time()
                    evaluate_times.append(evaluate_t_end-evaluate_t_start)
                    acc_total /= cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
                    loss_total /= cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
                    previous_accuracy = acc_total
                    print("Epoch: %f, Time: %f, Accuracy: %f, Loss: %f" % (cur_epoch, time.time() - sum(evaluate_times) - t_start, acc_total, loss_total))

                    if acc_total >= FLAGS.accuracy_to_reach:
                        break
                comm.Barrier()

            # Perform distributed gradient descent
            t_compute_start = time.time()
            materialized_gradients = []
            if rank != 0:
                fd = get_feed_dict(FLAGS.batch_size, images_train_raw, labels_train_raw, images, labels)
                materialized_gradients = sess.run([x[0] for x in grads_and_vars], feed_dict=fd)

                # Save gradients on a particular worker
                if rank == 1 and FLAGS.save_gradient_magnitude_histogram:
                    if iteration == 0:
                        print("Plotting gradient magnitude histograms")
                        plt.cla()
                        figs, axes = plt.subplots(nrows=2, ncols=5, figsize=(15*3,15))
                        axes = axes.flatten()
                        for i, (gradient, variable) in enumerate(zip(materialized_gradients, [x[1] for x in grads_and_vars])):
                            vname = variable.name.replace("/","_")
                            name = "iteration_%d_gradient_%s" % (iteration, vname)
                            title = "Layer %s" % vname
                            magnitudes = [abs(x) for x in list(gradient.flatten())]
                            axes[i].hist(magnitudes, bins='auto')
                            axes[i].set_title(title, fontsize=30)
                        figs.tight_layout()
                        plt.savefig("SparsifyHistogramOfGradientMagnitudes.png")
                        print("Done!")

            comm.Barrier()
            t_compute_end = time.time()
            compute_times += t_compute_end-t_compute_start

            t_accumulate_gradients_start = time.time()
            aggregate_and_apply_gradients(sess, model_variables, comm, rank, size, materialized_gradients, apply_gradients_placeholders, apply_gradients_op, previous_accuracy)
            t_accumulate_gradients_end = time.time()
            accumulate_gradients_times += t_accumulate_gradients_end-t_accumulate_gradients_start

            n_examples_processed += (size-1) * FLAGS.batch_size
            iteration += 1

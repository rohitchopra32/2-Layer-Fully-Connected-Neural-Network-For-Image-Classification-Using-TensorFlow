from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os.path
import data_helpers
import two_layer_neural_network

# Model parameters as external flags
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = 3072
CLASSES = 10

beginTime = time.time()

logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
print(logdir)


# Load CIFAR-10 data
data_sets = data_helpers.load_data()



# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS],
  name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

# Operation for the classifier's result
logits = two_layer_neural_network.inference(images_placeholder, IMAGE_PIXELS,
  FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)

# Define summery-operation for 'logits'-variable
tf.summary.histogram('logits', logits)

# Operation for the loss function
loss = two_layer_neural_network.loss(logits, labels_placeholder)

# Operation for the training step
train_step = two_layer_neural_network.training(loss, FLAGS.learning_rate)

# Operation calculating the accuracy of our predictions
accuracy = two_layer_neural_network.evaluation(logits, labels_placeholder)

# Operation merging summary data for TensorBoard
run_metadata = tf.RunMetadata()
# Define saver to save model state at checkpoints
saver = tf.train.Saver()


with tf.Session() as sess:
  # Initialize variables and create summary-writer
  summary = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(logdir, sess.graph)
  sess.run(tf.global_variables_initializer())
  # Generate input data batches
  zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
  batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size,
    FLAGS.max_steps)

  for i in range(FLAGS.max_steps):

    # Get next input data batch
    batch = next(batches)
    images_batch, labels_batch = zip(*batch)
    feed_dict = {
      images_placeholder: images_batch,
      labels_placeholder: labels_batch
    }

    # Periodically print out the model's current accuracy
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
      print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)
      summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
      summary_writer.add_summary(summary_str, i)

    # Perform a single training step
    sess.run([train_step, loss], feed_dict=feed_dict)

    # Periodically save checkpoint
    if (i + 1) % 1000 == 0:
      checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
      saver.save(sess, checkpoint_file, global_step=i)
      print('Saved checkpoint')

  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: data_sets['images_test'],
    labels_placeholder: data_sets['labels_test']})
  print('Test accuracy {:g}'.format(test_accuracy))

  summary_writer.close()
endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))


# subprocess.run("tensorboard --logdir=r'C:/Users/rohit/Desktop/shiwam project/%s"%logdir, shell=True)
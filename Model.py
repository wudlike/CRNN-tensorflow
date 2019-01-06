import tensorflow as tf
from tensorflow.contrib import rnn
from tfrecoder import read_tfrecord
import config
import os

class CRNN(object):
    def __init__(self, batch_size, max_width, init_learning_rate, dataset_path, epochs, checkpoint_dir):
        self.batch_size = batch_size
        self.max_width = max_width
        self.dataset_path = dataset_path
        self.num_classes = config.NUM_CLASS
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate=init_learning_rate, global_step=self.global_step,
                                                        decay_rate=0.9, decay_steps=500, staircase=True)

        self.softmax_w = tf.Variable(tf.truncated_normal(shape=[512, self.num_classes], mean=0, stddev=0.1),
                                     name='weight_w')
        self.softmax_b = tf.Variable(tf.constant(0.0, shape=[self.num_classes]), name='bias')

        #
        self.seq_len = tf.placeholder(shape=[None], dtype=tf.int32, name='seq_len')

        #
        (self.logits,
         self.max_char_count,
         self.optimizer,
         self.cost_loss,
         self.dense_decoded,
         self.error_rate,
         self.dense_label_batch) = self.progress()

    def CNN_VGG(self, inputs):
        ''' CNN extract feature from each input image, 网络架构选择的是VGG(CRNN)
        @param inputs: the input image
        @return: feature maps
        '''
        with tf.variable_scope('VGG_CNN'):
            conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, name='pool_1')

            conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, name='pool_2')

            conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_3')

            conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_4')
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(1, 2), strides=2, name='pool_3')

            conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_5')
            bn1 = tf.layers.batch_normalization(conv5, name='bn1')

            conv6 = tf.layers.conv2d(inputs=bn1, filters=512, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_6')
            bn2 = tf.layers.batch_normalization(conv6, name='bn_2')
            pool4 = tf.layers.max_pooling2d(inputs=bn2, pool_size=(1, 2), strides=2, name='pool_4')

            conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_7')
        return conv7

    def RNN(self, input, seq_len):
        with tf.variable_scope('BiLSTM_1'):
            lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
            lstm_bw_cell_1 = rnn.BasicLSTMCell(256)
            inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1,
                                                              lstm_bw_cell_1,
                                                              input, seq_len,
                                                              dtype=tf.float32)
            inter_output = tf.concat(inter_output, 2)
        with tf.variable_scope('BiLSTM_2'):
            lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
            lstm_bw_cell_2 = rnn.BasicLSTMCell(256)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2,
                                                         lstm_bw_cell_2,
                                                         inter_output, seq_len,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
        return outputs

    def progress(self):
        image_batch, label_batch = read_tfrecord(self.dataset_path, self.max_width, self.batch_size)
        # transfer the sparse vector label_batch to the dense vector -- the ground truth label
        dense_label_batch = tf.sparse_tensor_to_dense(label_batch)
        #
        cnn_output = self.CNN_VGG(image_batch)
        reshaped_cnn_output = tf.reshape(cnn_output, shape=[self.batch_size, -1, 512])
        max_char_count = reshaped_cnn_output.get_shape().as_list()[1]
        rnn_output = self.RNN(reshaped_cnn_output, self.seq_len)
        logits = tf.reshape(rnn_output, shape=[-1, 512])
        #
        logits = tf.matmul(logits, self.softmax_w) + self.softmax_b
        logits = tf.reshape(logits, shape=[self.batch_size, -1, self.num_classes])
        # final layer, the output of BLSTM
        logits = tf.transpose(logits, (1, 0, 2))
        # computer the CTC(Connectionist Temporal Classification) Loss
        loss = tf.nn.ctc_loss(labels=label_batch, inputs=logits, sequence_length=self.seq_len)
        cost_loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_loss, self.global_step)
        #
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        # tf.edit_distance()计算序列之间的编辑距离
        error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), label_batch))
        return logits, max_char_count, optimizer, cost_loss, dense_decoded, error_rate, dense_label_batch

    def index_to_word(self, result):
        '''
        @param result: a array contains the index of a word in the directory
        @return: the corresponding character in the directory
        '''
        return ''.join([config.CHAR_VECTOR[i] for i in result])

    def train(self):
        with tf.Session() as session:
            session.run(tf.group(tf.global_variables_initializer(),
                                 tf.local_variables_initializer()))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=session)

            print('start training')
            saver = tf.train.Saver()
            for index in range(self.epochs):
                _, loss, error_rate, decoded, dense_label_batch = session.run(
                    [self.optimizer, self.cost_loss, self.error_rate,
                     self.dense_decoded, self.dense_label_batch],
                    feed_dict={self.seq_len: [self.max_char_count] * self.batch_size})
                # print('loss:', loss, 'error_rate:', error_rate)
                if index % 100 == 0:
                    print('loss', loss, 'error_rate:', error_rate)
                    for predicted_index, truth_label in zip(decoded, dense_label_batch):
                        print('the prediction', self.index_to_word(predicted_index),
                              'the truth', self.index_to_word(truth_label))
                        print('predicted_index_length:', len(predicted_index),
                              'the truth:', len(truth_label))
                if index % 1000 == 0:
                    saver.save(session, os.path.join('./saver', 'model.ckpt'), global_step=index)
            coord.request_stop()
            coord.join(threads=threads)

    def test(self):
        saver = tf.train.Saver()
        with tf.Session() as session:
            print('testing')
            # check the checkpoint_dir and restore
            ckpt = tf.train.get_checkpoint_state('./saver')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            #
            session.run(tf.group(tf.global_variables_initializer(),
                                 tf.local_variables_initializer()))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=session)
            num_iter = int(config.NUM_EXAMPLES_PER_EPOCH_FOR_TEST / self.batch_size)
            presicion = 0.0
            for i in range(num_iter):
                decoded, error_rate, dense_label_batch= session.run([self.dense_decoded, self.error_rate, self.dense_label_batch],
                                                                  feed_dict={self.seq_len: [self.max_char_count] * self.batch_size})
                presicion = presicion + error_rate
                for predicted_index, truth_label in zip(decoded, dense_label_batch):
                    print('the prediction', self.index_to_word(predicted_index),
                          'the truth', self.index_to_word(truth_label))
            print('precision', presicion / (num_iter * self.batch_size))
            coord.request_stop()
            coord.join(threads=threads)



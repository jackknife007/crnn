# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import glob
from PIL import Image

TRAIN_PATH = './captcha3-10_train/*.png'
TEST_PATH = './captcha3-10_test/*.png'
CHAR_SET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
BATCH_SIZE = 16
HEIGHT = 64
MAX_WEIGHT = 300
NUM_CLASS = len(CHAR_SET) + 1


def get_image_and_text(path):
    text = path.split('\\')[-1].split('.')[0].split('_')[1]
    img = Image.open(path).convert('L')
    image_array = np.array(img).transpose() / 255
    _seq_len = len(image_array)
    return image_array, str(text), _seq_len


def get_sparse_tuple(sequence, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequence):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = list(map(CHAR_SET.index, values))
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequence), indices.max(0)[1]+1], dtype=np.int64)
    return indices, values, shape


def get_next_batch():
    train_data = glob.glob(TRAIN_PATH)
    train_len = len(train_data)
    i = 0
    while True:
        if not i < train_len:
            i = 0
            np.random.shuffle(train_data)
        _inputs = np.zeros([BATCH_SIZE, MAX_WEIGHT, HEIGHT])
        _seq_len = []
        _labels = []
        for n, sample in enumerate(train_data[i:i+BATCH_SIZE]):
            image_array, _text, __seq_len = get_image_and_text(sample)
            _labels.append(_text)
            _inputs[n, :__seq_len] = image_array
            _seq_len.append(np.floor(__seq_len / 4 - 3))
        i = i + BATCH_SIZE
        yield np.reshape(_inputs, [BATCH_SIZE, MAX_WEIGHT, HEIGHT, 1]), get_sparse_tuple(_labels), _seq_len


def get_test_data(size=1):
    test_data = glob.glob(TEST_PATH)
    _inputs = np.zeros([size, MAX_WEIGHT, HEIGHT])
    _seq_len = []
    _labels = []
    for n, sample in enumerate(np.random.choice(test_data, size, False)):
        image_array, _text, __seq_len = get_image_and_text(sample)
        _labels.append(_text)
        _inputs[n, :__seq_len] = image_array
        _seq_len.append(np.floor(__seq_len / 4 - 3))
    return np.reshape(_inputs, [size, MAX_WEIGHT, HEIGHT, 1]), _labels, _seq_len


def build_graph():
    _inputs = tf.placeholder(tf.float32, [None, MAX_WEIGHT, HEIGHT, 1])
    _labels = tf.sparse_placeholder(tf.int32)
    _seq_len = tf.placeholder(tf.int32, [None])

    # CNN
    conv1 = tf.layers.conv2d(inputs=_inputs, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])
    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2])
    conv5 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu)
    bn1 = tf.layers.batch_normalization(conv5)
    conv6 = tf.layers.conv2d(inputs=bn1, filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu)
    bn2 = tf.layers.batch_normalization(conv6)
    pool4 = tf.layers.max_pooling2d(inputs=bn2, pool_size=[2, 2], strides=[1, 2])
    conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[2, 2], strides=[1, 1], padding='valid', activation=tf.nn.relu)

    # to sequence
    _shape = conv7.get_shape().as_list()
    feature_num = _shape[2] * _shape[3]
    seq = tf.reshape(conv7, [-1, feature_num])
    w1 = tf.Variable(tf.truncated_normal([feature_num, 512], stddev=0.1))
    b1 = tf.Variable(tf.constant(0., shape=[512]))
    seq = tf.reshape(tf.matmul(seq, w1)+b1, [-1, _shape[1], 512])

    # bRNN
    with tf.variable_scope("lstm_1"):
        lstm_fw_cell_1 = tf.nn.rnn_cell.LSTMCell(256)
        lstm_bw_cell_1 = tf.nn.rnn_cell.LSTMCell(256)
        output_1, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, seq, _seq_len, dtype=tf.float32)
        output_1 = tf.concat(output_1, 2)

    with tf.variable_scope("lstm_2"):
        lstm_fw_cell_2 = tf.nn.rnn_cell.LSTMCell(256)
        lstm_bw_cell_2 = tf.nn.rnn_cell.LSTMCell(256)
        output_2, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, output_1, _seq_len, dtype=tf.float32)
        output_2 = tf.concat(output_2, 2)

    # res
    output_2 = tf.add(output_2, seq)

    # fc
    fc_in = tf.reshape(output_2, [-1, 512])
    w2 = tf.Variable(tf.truncated_normal([512, NUM_CLASS], stddev=0.1))
    b2 = tf.Variable(tf.constant(0., shape=[NUM_CLASS]))

    logits = tf.reshape(tf.matmul(fc_in, w2)+b2, [-1, _shape[1], NUM_CLASS])

    # time_major
    _logits = tf.transpose(logits, [1, 0, 2])

    return _inputs, _labels, _logits, _seq_len


def get_acc(ground_truth, predict):
    acc = 0
    num = len(ground_truth)
    for i in range(num):
        _predict = ''.join([CHAR_SET[c] for c in predict[i] if not c == -1])
        if _predict == ground_truth[i]:
            acc = acc + 1
    return acc/num


def train(max_step=100000):
    inputs, labels, logits, seq_len = build_graph()
    loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=seq_len))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
    decode, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len, NUM_CLASS, merge_repeated=False)
    dense_decode = tf.sparse_tensor_to_dense(decode[0], default_value=-1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Training..........')
        sess.run(tf.global_variables_initializer())
        step = 0
        get_batch = get_next_batch()
        while step < max_step:
            train_inputs, train_labels, train_seq_len = next(get_batch)
            _, _loss = sess.run([optimizer, loss], feed_dict={inputs: train_inputs, labels: train_labels, seq_len: train_seq_len})
            if step % 100 == 0:
                test_inputs, test_labels, test_seq_len = get_test_data(100)
                _res = sess.run(dense_decode, feed_dict={inputs: test_inputs, seq_len: test_seq_len})
                acc = get_acc(test_labels, _res)
                print(step, 'loss: %f, accuracy: %f' % (_loss, acc))
                if acc > 0.85:
                    saver.save(sess, 'ckpt/crnn.ckpt', global_step=step)
            step = step + 1


def test(num=20):
    print('-'*50)
    print('Testing..........')
    inputs, labels, logits, seq_len = build_graph()
    decode, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len, NUM_CLASS, merge_repeated=False)
    dense_decode = tf.sparse_tensor_to_dense(decode[0], default_value=-1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('ckpt/'))
        test_inputs, test_labels, test_seq_len = get_test_data(num)
        _res = sess.run(dense_decode, feed_dict={inputs: test_inputs, seq_len: test_seq_len})
        for i in range(num):
            print('True: %s, Predict: %s ' % (test_labels[i], ''.join([CHAR_SET[c] for c in _res[i] if not c == -1])))


if __name__ == "__main__":
    train(300000)
    #test(20)


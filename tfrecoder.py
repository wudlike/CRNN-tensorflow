# coding=utf-8
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random
from config import CHAR_VECTOR
from config import NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
from config import NUM_EXAMPLES_PER_EPOCH_FOR_TEST

VOCUBLARY_SIZE = len(CHAR_VECTOR)


def resize_image(image):
    '''resize the size of image'''
    width, height = image.size
    ratio = 32.0 / float(height)
    image = image.resize((int(width * ratio), 32))
    return image


def generation_vocublary(CHAR_VECTOR):
    vocublary = {}
    index = 0
    for char in CHAR_VECTOR:
        vocublary[char] = index
        index = index + 1
    return vocublary


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generation_TFRecord(data_dir):
    vocublary = generation_vocublary(CHAR_VECTOR)

    image_name_list = []
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            image_name_list.append(file)

    random.shuffle(image_name_list)
    capacity = len(image_name_list)

    # 生成train tfrecord文件
    train_writer = tf.python_io.TFRecordWriter('G:\\图像文字识别\\Train_en_10000\\train_dataset.tfrecords')
    train_image_name_list = image_name_list[0:int(capacity * 0.9)]
    for train_name in train_image_name_list:
        train_image_label = []
        for s in train_name.strip('.jpg'):
            if s == '_':
                break
            if s == ' ':
                continue
            train_image_label.append(vocublary[s])

        train_image = Image.open(os.path.join(data_dir, train_name))
        train_image = resize_image(train_image)
        # print(image.size)
        train_image_array = np.asarray(train_image, np.uint8)
        train_shape = np.array(train_image_array.shape, np.int32)
        train_image = train_image.tobytes()

        train_example = tf.train.Example(features=tf.train.Features(feature={
            'label': int64_list_feature(train_image_label),
            'image': bytes_feature(train_image),
            'h': int64_feature(train_shape[0]),
            'w': int64_feature(train_shape[1]),
            'c': int64_feature(train_shape[2])
        }))
        train_writer.write(train_example.SerializeToString())
    train_writer.close()

    # 生成test tfrecord文件
    test_writer = tf.python_io.TFRecordWriter('G:\\图像文字识别\\Train_en_10000\\test_dataset.tfrecords')
    test_image_name_list = image_name_list[int(capacity * 0.9):capacity]
    for test_name in test_image_name_list:
        test_image_label = []
        for s in test_name.strip('.jpg'):
            if s == '_':
                break
            if s == '':
                continue
            test_image_label.append(vocublary[s])

        test_image = Image.open(os.path.join(data_dir, test_name))
        test_image = resize_image(test_image)
        # print(image.size)
        test_image_array = np.asarray(test_image, np.uint8)
        test_shape = np.array(test_image_array.shape, np.int32)
        test_image = test_image.tobytes()

        test_example = tf.train.Example(features=tf.train.Features(feature={
            'label': int64_list_feature(test_image_label),
            'image': bytes_feature(test_image),
            'h': int64_feature(test_shape[0]),
            'w': int64_feature(test_shape[1]),
            'c': int64_feature(test_shape[2])
        }))
        test_writer.write(test_example.SerializeToString())
    test_writer.close()


def read_tfrecord(filename, max_width, batch_size, train=True):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialize_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(serialized=serialize_example,
                                             features={
                                                 # 'label': tf.FixedLenFeature([VOCUBLARY_SIZE], tf.int64),
                                                 'label': tf.VarLenFeature(dtype=tf.int64),
                                                 'image': tf.FixedLenFeature([], tf.string),
                                                 'h': tf.FixedLenFeature([], tf.int64),
                                                 'w': tf.FixedLenFeature([], tf.int64),
                                                 'c': tf.FixedLenFeature([], tf.int64)
                                             })
    h = tf.cast(image_features['h'], tf.int32)
    w = tf.cast(image_features['w'], tf.int32)
    c = tf.cast(image_features['c'], tf.int32)

    image = tf.decode_raw(image_features['image'], tf.uint8)
    # 将图片变成tensor，对图片进行归一化操作，将[0-255]之间的像素归一化到[-0.5, 0.5]
    # 标准化处理可以使得不同的特征具有相同的尺度， 这样在使用梯度下降法学参数时，不同特征对
    # 参数的影响程度就一样了
    # image = tf.cast(image, tf.float32)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, shape=[h, w, c])
    resized_image = tf.image.resize_image_with_crop_or_pad(image, target_height=32, target_width=max_width)
    resized_image = tf.reshape(resized_image, shape=[32, max_width, 3])

    label = tf.cast(image_features['label'], tf.int32)

    min_fraction_of_example_in_queue = 0.4
    if train is True:
        min_queue_examples = int(min_fraction_of_example_in_queue * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        train_image_batch, train_label_batch = tf.train.shuffle_batch([resized_image, label],
                                                                      batch_size=batch_size,
                                                                      capacity=min_queue_examples + 3 * batch_size,
                                                                      min_after_dequeue=min_queue_examples,
                                                                      num_threads=32)
        return train_image_batch, train_label_batch
    else:
        min_queue_examples = int(min_fraction_of_example_in_queue * NUM_EXAMPLES_PER_EPOCH_FOR_TEST)
        test_image_batch, test_label_batch = tf.train.batch([resized_image, label],
                                                            batch_size=batch_size,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            num_threads=32)
        return test_image_batch, test_label_batch

def index_to_word(result):
    return ''.join([CHAR_VECTOR[i] for i in result])

def main(argv):
    generation_TFRecord('G:\\图像文字识别\\Train_en_10000\\train')
    # # generation_TFRecord('./dataset/images')
    # # train_image, train_label = read_tfrecord('./dataset/train_dataset.tfrecords', 250, 32)
    # test_image, test_label = read_tfrecord('G:\\图像文字识别\\Train_en_10000\\test_dataset.tfrecords', 1000, 32)
    # test_label = tf.sparse_tensor_to_dense(test_label)
    # with tf.Session() as session:
    #     session.run(tf.group(tf.global_variables_initializer(),
    #                          tf.local_variables_initializer()))
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     # image_train, label_train = session.run([train_image, train_label])
    #     # print(image_train.shape)
    #     image_test, label_test = session.run([test_image, test_label])
    #     print(image_test.shape)
    #     for label in label_test:
    #         print(len(label))
    #         print(index_to_word(label))
    #     coord.request_stop()
    #     coord.join(threads=threads)

if __name__ == '__main__':
    tf.app.run()

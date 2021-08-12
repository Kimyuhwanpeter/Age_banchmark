# -*- coding: utf-8 -*-
# DEX: Deep EXpectation of apparent age from a single image
from random import shuffle
from absl import app
from absl import flags
from DEX_model_2x import *

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import datetime

flags.DEFINE_integer('img_size', 224, 'Size of rows and cols (Image)')

flags.DEFINE_integer('channel', 3, 'Channel of image (RGB-3, gray-3)')

flags.DEFINE_integer('batch_size', 64, 'Batch size in training')

flags.DEFINE_integer('num_classes', 48, 'Number of classes')

flags.DEFINE_float('lr', 0.0001, 'Learning rate')

flags.DEFINE_string('img_path', '/yuwhan/yuwhan/Dataset/[1]Third_dataset/UTK/UTKFace/', 'Image directory')

flags.DEFINE_string('txt_path', '/yuwhan/yuwhan/Dataset/[2]Fourth_dataset/age_banchmark/train_data/UTK/train.txt', 'Text (with label information) directory')

flags.DEFINE_string('val_img_path', '/yuwhan/yuwhan/Dataset/[1]Third_dataset/UTK/UTKFace/', 'Validate image path')

flags.DEFINE_string('val_txt_path_1', '/yuwhan/yuwhan/Dataset/[2]Fourth_dataset/age_banchmark/train_data/UTK/test.txt', 'Validate text path')

flags.DEFINE_string('val_txt_path_2', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_69_fullDB/[3]MegaAge_43_69_and_Morph_16_42/MORPH_test_16_42.txt', 'Validate text path')

flags.DEFINE_integer('val_batch_size', 15, "Validation batch size")

flags.DEFINE_integer('val_batch_size_2', 128, "Validation batch size")

flags.DEFINE_bool('load_checkpoint', False, 'Load checkpoint')

flags.DEFINE_string('pre_checkpoint_path', '', 'Load checkpoint')

flags.DEFINE_string('save_checkpoint', '/yuwhan/Edisk/yuwhan/Edisk/4th_paper/age_banchmark/UTK/checkpoint', 'Checkpoint saving path')

flags.DEFINE_string("graphs", "/yuwhan/Edisk/yuwhan/Edisk/4th_paper/age_banchmark/UTK/graphs/", "")

flags.DEFINE_bool('Train', True, 'Train-True, Test-False')

flags.DEFINE_integer('total_epoch', 100, 'Total epoch')

flags.DEFINE_string("output_loss_txt", "/yuwhan/Edisk/yuwhan/Edisk/4th_paper/age_banchmark/UTK/loss.txt", "")

####################################################################################################
flags.DEFINE_string('test_img_path', 'C:/Users/Yuhwan/Desktop/morph/test1_re/', 'Test image path')

flags.DEFINE_string('test_txt_path', 'C:/Users/Yuhwan/Desktop/morph/test1_re.txt', 'Test Text path')
####################################################################################################

FLAGS = flags.FLAGS

optimizer = tf.keras.optimizers.Adam(0.0001)


def _func(filename, label):

    image_string = tf.io.read_file(filename)
    decode_image = tf.image.decode_jpeg(image_string, channels=3)
    decode_image = tf.image.resize(decode_image, [FLAGS.img_size, FLAGS.img_size])
    decode_image = tf.image.per_image_standardization(decode_image)

    label = label - 16
    label = tf.one_hot(label, FLAGS.num_classes)

    return decode_image, label

def _test_func(image, label):

    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)
    
    label = int(label) - 16

    return img, label

@tf.function
def train(images, labels, model):
    with tf.GradientTape() as tape:
        logits = run_model(model, images)
        total_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

@tf.function
def test(image, age_class, model):

    logits = model(images, training=False)
    logits = tf.nn.softmax(logits)
    
    predict_age = tf.reduce_sum(logits[0] * age_class)

    return predict_age

@tf.function
def val_cal(images, labels, age_class, model):

    logits = model(images, training=False)
    logits = tf.nn.softmax(logits, -1)
    
    pre_age = tf.reduce_sum(logits * age_class, 1) - 1.
    labels = tf.cast(labels, tf.float32)

    MAE = tf.reduce_sum(tf.math.abs(labels - pre_age))

    return MAE

@tf.function
def run_model(model, images, training=True):
    logits = model(images, training=True)
    return logits

def main(argv=None):
    ######################################################################################################
    train_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                       input_tensor=None, input_shape=(FLAGS.img_size, FLAGS.img_size, 3), pooling=None)
    regularizer = tf.keras.regularizers.l2(0.00005)

    pre_train_fc = tf.keras.applications.VGG16(include_top=True, weights='imagenet',
                       input_tensor=None, input_shape=(FLAGS.img_size, FLAGS.img_size, 3), pooling=None)

    for layer in train_model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    x = train_model.output
    h = tf.keras.layers.Flatten()(x)
    h = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005), name="dex_fc1")(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    h = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005), name="dex_fc2")(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    y = tf.keras.layers.Dense(FLAGS.num_classes, name='last_layer')(h)

    train_model = tf.keras.Model(inputs=train_model.input, outputs=y)

    train_model.get_layer("dex_fc1").set_weights(pre_train_fc.get_layer("fc1").get_weights())
    train_model.get_layer("dex_fc2").set_weights(pre_train_fc.get_layer("fc2").get_weights())

    train_model.summary()
    ######################################################################################################
    #train_my_model = MY_VGG_16(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.channel))
    #train_model.summary()
    ######################################################################################################

    # Load checkpoint
    if FLAGS.load_checkpoint == True:
        ckpt = tf.train.Checkpoint(train_model=train_model,optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    count = 0
    if FLAGS.Train == True:

        ################################################################################
        tr_data_name = np.loadtxt(FLAGS.txt_path, dtype='<U100', skiprows=0, usecols=0)
        tr_data_name = [FLAGS.img_path + data_name_ for data_name_ in tr_data_name]
        tr_data_label = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)
        ################################################################################

        ################################################################################
        age_class = np.arange(1,49).astype(np.float32)

        val_data_name = np.loadtxt(FLAGS.val_txt_path_1, dtype='<U100', skiprows=0, usecols=[0, 1, 2, 3])
        print(len(val_data_name))
        WM_img, WM_age = [], []
        WF_img, WF_age = [], []
        BM_img, BM_age = [], []
        BF_img, BF_age = [], []
        for i in range(len(val_data_name)):

            if val_data_name[i][2] == "M" and val_data_name[i][3] == "W":
                WM_img.append(FLAGS.val_img_path + val_data_name[i][0])
                WM_age.append(val_data_name[i][1])

            if val_data_name[i][2] == "F" and val_data_name[i][3] == "W":
                WF_img.append(FLAGS.val_img_path + val_data_name[i][0])
                WF_age.append(val_data_name[i][1])

            if val_data_name[i][2] == "M" and val_data_name[i][3] == "B":
                BM_img.append(FLAGS.val_img_path + val_data_name[i][0])
                BM_age.append(val_data_name[i][1])

            if val_data_name[i][2] == "F" and val_data_name[i][3] == "B":
                BF_img.append(FLAGS.val_img_path + val_data_name[i][0])
                BF_age.append(val_data_name[i][1])
        
        print(len(WM_img), len(WF_img), len(BM_img), len(BF_img))
        WM_img, WM_age = np.array(WM_img), np.array(WM_age)
        WF_img, WF_age = np.array(WF_img), np.array(WF_age)
        BM_img, BM_age = np.array(BM_img), np.array(BM_age)
        BF_img, BF_age = np.array(BF_img), np.array(BF_age)
        all_val_list = [[WM_img, WM_age], [WF_img, WF_age], [BM_img, BM_age], [BF_img, BF_age]]
       
        ################################################################################

        batch_idx = len(tr_data_name) // FLAGS.batch_size

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        loss_f = open(FLAGS.output_loss_txt, "w")

        for epoch_step in range(FLAGS.total_epoch):
            A = list(zip(tr_data_name, tr_data_label))
            shuffle(A)
            tr_data_name, tr_data_label = zip(*A)
            tr_data_name, tr_data_label = np.array(tr_data_name), np.array(tr_data_label)

            data_generator = tf.data.Dataset.from_tensor_slices((tr_data_name, tr_data_label))
            data_generator = data_generator.shuffle(len(tr_data_name))
            data_generator = data_generator.map(_func)
            data_generator = data_generator.batch(FLAGS.batch_size)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(data_generator)

            for step in range(batch_idx):

                batch_images, batch_labels = next(it)

                Loss = train(batch_images, batch_labels, train_model)
                if count % 10 == 0:
                    print('Epoch: {} [step: {} / total_step: {}] Loss = {}'.format(epoch_step + 1,
                                                                                   step + 1,
                                                                                   batch_idx + 1,
                                                                                   Loss))
                with train_summary_writer.as_default():
                    tf.summary.scalar(u'total loss', Loss, step=count)

                if count % 100 == 0:
                    test_list = ["WM", "WF", "BM", "BF"]
                    for j in range(len(all_val_list)):  # WM, WF, BM, BF
                        val_img, val_lab = all_val_list[j]

                        val_data_generator = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
                        val_data_generator = val_data_generator.map(_test_func)
                        val_data_generator = val_data_generator.batch(1)
                        val_data_generator = val_data_generator.prefetch(tf.data.experimental.AUTOTUNE)

                        val_idx = len(val_img) // 1
                        val_it = iter(val_data_generator)
                        AE = 0;
                        
                        for i in range(val_idx):
                            img, lab = next(val_it)
                            pre_age = val_cal(img, lab, age_class, train_model) # 이 함숙 고쳐야함
                            AE += pre_age

                        print("MAE = {} ({})".format(AE / len(val_img), test_list[j]))  # 왜 똑같이나오는가? 측정 함수 코드 문제
                        
                        loss_f.write("Epochs: {}, step = {}".format(epoch_step, count))
                        loss_f.write(" --> ")
                        loss_f.write(test_list[j])
                        loss_f.write(": ")
                        loss_f.write(str(AE / len(val_img)))
                        loss_f.write(", ")

                    loss_f.write("\n")
                    loss_f.flush()  # 모든 실험할 age estimation 코드에서 지금 이 부분을 추가로 넣을것
                   
                    # model_dir = FLAGS.save_checkpoint
                    # folder_name = int(count//100)
                    # folder_neme_str = '%s/%s' % (model_dir, folder_name)
                    # if not os.path.isdir(folder_neme_str):
                    #     print("Make {} folder to save checkpoint".format(folder_name))
                    #     os.makedirs(folder_neme_str)
                    # checkpoint = tf.train.Checkpoint(train_model=train_model,optimizer=optimizer)
                    # checkpoint_dir = folder_neme_str + "/" + "DEX_{}_steps.ckpt".format(count + 1)
                    # checkpoint.save(checkpoint_dir)

                count += 1

    else:
        age_class = np.arange(1,51).astype(np.float32)

        data_name = np.loadtxt(FLAGS.test_txt_path, dtype='<U100', skiprows=0, usecols=0)
        data_name = [FLAGS.test_img_path + data_name_ for data_name_ in data_name]
        data_label = np.loadtxt(FLAGS.test_txt_path, dtype=np.float32, skiprows=0, usecols=1)

        data_generator = tf.data.Dataset.from_tensor_slices((data_name, data_label))
        data_generator = data_generator.map(_test_func)
        data_generator = data_generator.batch(1)
        data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

        it = iter(data_generator)
        AE = 0.
        for i in range(len(data_label)):

            image, age = next(it)

            predice_age = val_cal(image, age, age_class, train_model)

            AE += predice_age

            if i % 100 == 0:
                print('MAE about {} images... = {}'.format(i + 1, AE / (i + 1)))

        print('========================')
        print('MAE = {}'.format(AE / len(data_label)))
        print('========================')

if __name__ == '__main__':
    app.run(main)

# -*- coding: utf-8 -*-
# https://github.com/Raschka-research-group/coral-cnn/tree/master/model-code/resnet34
from absl import flags, app
from Rank_consistent_model_fix import *
from Rank_consistent_model import *
from random import shuffle, random

import tensorflow as tf
import numpy as np
# import cv2
import os
import sys
import datetime

flags.DEFINE_string('img_path', '/yuwhan/yuwhan/Dataset/[1]Third_dataset/UTK/UTKFace/', 'Image directory')

flags.DEFINE_string('txt_path', '/yuwhan/yuwhan/Dataset/[2]Fourth_dataset/age_banchmark/train_data/UTK/train.txt', 'Text (with label information) directory')

flags.DEFINE_string('val_img_path', '/yuwhan/yuwhan/Dataset/[1]Third_dataset/UTK/UTKFace/', 'Validate image path')

flags.DEFINE_string('val_txt_path', '/yuwhan/yuwhan/Dataset/[2]Fourth_dataset/age_banchmark/train_data/UTK/test.txt', 'Validate text path')

flags.DEFINE_string("val_txt_path_2", "D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_69_fullDB/[1]Full_DB/testB.txt", "Validataion text path")

flags.DEFINE_integer('img_size', 128, 'Image size')

flags.DEFINE_integer('ch', 3, 'Image channels')

flags.DEFINE_integer('batch_size', 256, 'Train Batch size')

flags.DEFINE_integer("val_batch_size", 128, "Validation Batch size")

flags.DEFINE_integer("val_batch_size_2", 128, "Validation2 batch size")

flags.DEFINE_integer('num_classes', 48, 'Number of classes')

flags.DEFINE_integer('epochs', 5000, 'Total epochs of training')

flags.DEFINE_float("lr", 5e-5, "Learning rate")

flags.DEFINE_string('weights', "/yuwhan/yuwhan/Projects/[1]Age_related_work_2.x_My_version/Rank-consistent Ordinal Regression for Neural/resnet34_imagenet_1000_no_top.h5", '')

flags.DEFINE_bool('train', True, 'True or False')

flags.DEFINE_bool('pre_checkpoint', False, 'True or False')

flags.DEFINE_string('pre_checkpoint_path', '', 'Saved checkpoint path')

flags.DEFINE_string('save_checkpoint', '', 'Save checkpoint path')

flags.DEFINE_string("graphs", "", "")

flags.DEFINE_integer('n_test', 10000, 'Number of test images')

flags.DEFINE_string('test_txt', '', 'Test text(label) path')

flags.DEFINE_string('test_img', '', 'Test images path')

flags.DEFINE_string("output_loss_txt", "/yuwhan/Edisk/yuwhan/Edisk/4th_paper/age_banchmark/UTK/loss_CORAL.txt", "")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optimizer = tf.keras.optimizers.Adam(FLAGS.lr,beta_1=0.9, beta_2=0.99)


def _func(filename, label):

    image_string = tf.io.read_file(filename)
    decode_image = tf.image.decode_jpeg(image_string, channels=3)
    decode_image = tf.image.resize(decode_image, [FLAGS.img_size - 8, FLAGS.img_size - 8])  / 255.
    #decode_image = tf.image.random_crop(decode_image, [FLAGS.img_size - 8, FLAGS.img_size - 8, 3])

    if random() > 0.5:
        decode_image = tf.image.flip_left_right(decode_image)

    #decode_image = tf.image.per_image_standardization(decode_image)

    label = label - 16
    one_hot = tf.one_hot(label, FLAGS.num_classes)
    
    return decode_image, one_hot, label

def val_func(name, label):

    image_string = tf.io.read_file(name)
    decode_image = tf.image.decode_jpeg(image_string, channels=3)
    decode_image = tf.image.resize(decode_image, [FLAGS.img_size - 8, FLAGS.img_size - 8]) / 255.
    #decode_image = tf.image.per_image_standardization(decode_image)

    label = int(label) - 16
    one_hot = tf.one_hot(label, FLAGS.num_classes)  
    
    return decode_image, one_hot

#@tf.function
def run_model(model, images):
    logits, probs = model(images, training=True)
    return logits, probs

@tf.function
def train_step(model, images, levels, imp):
    
    with tf.GradientTape() as tape:
        logits, probs = run_model(model, images)

        #total_loss = (-tf.reduce_sum((tf.nn.log_softmax(logits, axis=2)[:,:,1]*levels + tf.nn.log_softmax(logits, axis=2)[:,:,0]*(1-levels))*imp, 1))
                        
        # total_loss = (-tf.reduce_sum( (tf.math.log_sigmoid(logits)*levels + tf.math.log(1. - tf.nn.sigmoid(logits))*(1-levels))*imp, 1))
        total_loss = (-tf.reduce_sum( (tf.math.log_sigmoid(logits)*levels + (tf.math.log_sigmoid(logits) - logits)*(1-levels))*imp, 1))
        #total_loss = -tf.reduce_sum((tf.math.log(tf.nn.softmax(logits, 2)[:, :, 1] + 1e-7) * levels \
        #    + tf.math.log(tf.nn.softmax(logits, 2)[:, :, 0] + 1e-7) * (1 - levels)) * imp, 1)

        total_loss = tf.reduce_mean(total_loss)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

def task_importance_weights(data):
    label = np.array(data).astype(np.float32)
    num_examples = label.size

    y = np.unique(label)
    
    m = np.zeros(label.shape)

    for i, t in enumerate(np.arange(np.min(y), np.max(y))):
        m_k = np.max([label[label > t].size, 
                        num_examples - label[label > t].size])
        #print(m_k)
        m_k = tf.cast(tf.convert_to_tensor(m_k), tf.float32)
        m[i] = tf.sqrt(m_k)
        # m[i] = float(m_k)**(0.5)

    max_ = np.max(m)
    imp = tf.cast(m / max_, tf.float32)
    #print(imp)
    return imp
    
@tf.function
def test_MAE(model, images, labels):
    logits, probs = model(images, training=False)
    predict = probs > 0.5
    predict = tf.cast(predict, tf.float32)
    pre_age = tf.reduce_sum(predict, 1)
    grd_age = tf.argmax(labels, 1) + 1
    grd_age = tf.cast(grd_age, tf.float32)
    AE = tf.reduce_sum(tf.math.abs(grd_age - pre_age))
    return AE

def make_levels(labels):
    levels = []
    for i in range(FLAGS.batch_size):
        l = [1] * (labels[i].numpy()) + [0]*(FLAGS.num_classes - 1 - labels[i].numpy())
        l = tf.cast(l, tf.float32)
        levels.append(l)

    return tf.convert_to_tensor(levels, tf.float32)

def main(argv=None):

    # train_model = resnet_type1(input_shape=(FLAGS.img_size - 8, FLAGS.img_size - 8, 3), NUM_CLASSES=FLAGS.num_classes)
    train_model = ResNet34(input_shape=(FLAGS.img_size - 8, FLAGS.img_size - 8, FLAGS.ch), include_top=False,
                        batch_size=FLAGS.batch_size, weight_path=FLAGS.weights, weights='imagenet')

    regularizer = tf.keras.regularizers.l2(0.000005)
    initializer = tf.keras.initializers.glorot_normal()

    for layer in train_model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
        # for attr_ in ["kernel_initializer"]:
        #     if hasattr(layer, attr_):
        #         setattr(layer, attr_, initializer)

    x = train_model.output
    avgpool = tf.keras.layers.GlobalAveragePooling2D()(x)
    # avgpool = tf.reshape(avgpool, [avgpool.shape[0], -1])
    # fc = tf.keras.layers.Dense(1, use_bias=False)(avgpool)

    # logits = Linear(NUM_CLASSES - 1)(fc)
    logits = tf.keras.layers.Dense(FLAGS.num_classes-1, use_bias=False)(avgpool)
    logits = Linear(FLAGS.num_classes - 1)(logits)
    probs = tf.nn.sigmoid(logits)

    train_model = tf.keras.Model(inputs=train_model.input, outputs=[logits, probs])
    train_model.summary()

    #for m in train_model.layers:
    #    if isinstance(m, tf.keras.layers.Conv2D):
    #        a = m.output_mask
    #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #        m.weight.data.normal_(0, (2. / n)**.5)
    #    elif isinstance(m, tf.keras.layers.BatchNormalization):
    #        m.get_weights
    #        m.weight.data.fill_(1)
    #        m.bias.data.zero_()

    if FLAGS.pre_checkpoint is True:

        ckpt = tf.train.Checkpoint(train_model=train_model, optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            print(ckpt_manager.latest_checkpoint)
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    if FLAGS.train == True:
    
        data_name = np.loadtxt(FLAGS.txt_path, dtype='<U100', skiprows=0, usecols=0)
        data_name = [FLAGS.img_path + data_name_ for data_name_ in data_name]
        data_label = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        imp = task_importance_weights(data_label-16)
        imp = imp[0:FLAGS.num_classes-1]

        val_data_name = np.loadtxt(FLAGS.val_txt_path, dtype='<U100', skiprows=0, usecols=[0, 1, 2, 3])
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

        batch_idx = len(data_label) // FLAGS.batch_size

        #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #train_log_dir = FLAGS.graphs + current_time + '/train'
        #val_log_dir = FLAGS.graphs + current_time + '/val'
        #train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        #val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        loss_f = open(FLAGS.output_loss_txt, "w")
        count = 0
        for epoch in range(FLAGS.epochs):

            A = list(zip(data_name, data_label))
            shuffle(A)
            data_name, data_label = zip(*A)
            data_name = np.array(data_name)
            data_label = np.array(data_label)

            data_generator = tf.data.Dataset.from_tensor_slices((data_name, data_label))
            data_generator = data_generator.shuffle(len(data_name))
            data_generator = data_generator.map(_func)
            data_generator = data_generator.batch(FLAGS.batch_size)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(data_generator)

            #imp = task_importance_weights(data_label)
            #imp = imp[0:FLAGS.num_classes-1]
            for step in range(batch_idx):

                batch_images, batch_labels, age = next(it)
                levels = make_levels(age)
                total_loss = train_step(train_model, batch_images, levels, imp)

                #with val_summary_writer.as_default():
                #    tf.summary.scalar(u'total loss', loss, step=count)

                if count % 10 == 0:
                    #MAE = test_MAE(train_model, batch_images, batch_labels, levels)
                    print('Epoch: {} [{}/{}] loss = {}'.format(epoch + 1, step + 1, batch_idx, total_loss))

                if count % 100 == 0:
                    test_list = ["WM", "WF", "BM", "BF"]
                    for j in range(len(all_val_list)):
                        val_img, val_lab = all_val_list[j]

                        val_data_generator = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
                        val_data_generator = val_data_generator.map(val_func)
                        val_data_generator = val_data_generator.batch(1)
                        val_data_generator = val_data_generator.prefetch(tf.data.experimental.AUTOTUNE)

                        val_idx = len(val_img) // 1
                        val_it = iter(val_data_generator)
                        AE = 0

                        for i in range(val_idx):
                            img, lab = next(val_it)
                            pre_age = test_MAE(train_model, img, lab)
                            AE += pre_age

                        print("MAE = {} ({})".format(AE / len(val_img), test_list[j]))

                        loss_f.write("Epochs: {}, step = {}".format(epoch, count))
                        loss_f.write(" --> ")
                        loss_f.write(test_list[j])
                        loss_f.write(": ")
                        loss_f.write(str(AE / len(val_img)))
                        loss_f.write(", ")

                    loss_f.write("\n")
                    loss_f.flush()



                #    print("==========")
                #    print("[2]MAE = {}".format(MAE))
                #    print("==========")
                #    model_dir = FLAGS.save_checkpoint
                #    folder_name = int((count + 1)/val_idx)
                #    folder_name_str = "%s/%s" % (model_dir, folder_name)
                #    if not os.path.isdir(folder_name_str):
                #        print("Make {} folder to save checkpoint".format(folder_name))
                #        os.makedirs(folder_name_str)
                #    ckpt = tf.train.Checkpoint(train_model=train_model, optimizer=optimizer)
                #    checkpoint_dir = folder_name_str + "/" + "CORAL_{}_steps.ckpt".format(count)
                #    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, 5)
                #    ckpt_manager.save()

                #    with val_summary_writer.as_default():
                #        tf.summary.scalar(u'[2]MAE', MAE, step=count)

                count += 1

    else:
        data_name = np.loadtxt(FLAGS.test_txt, dtype='<U100', skiprows=0, usecols=0)
        data_name = [FLAGS.test_img + data_name_ for data_name_ in data_name]
        data_label = np.loadtxt(FLAGS.txt_path, dtype=np.float32, skiprows=0, usecols=1)

        data_generator = tf.data.Dataset.from_tensor_slices((data_name, data_label))
        data_generator = data_generator.shuffle(len(data_name))
        data_generator = data_generator.map(_func)
        data_generator = data_generator.batch(1)
        data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

        MAE = 0
        it = iter(data_generator)
        for i in range(FLAGS.n_test):

            image, labels, opp_labels = next(it)

            _, probs = train_model(image, training=False)

            predict = probs > 0.5
            predict = tf.cast(predict, tf.float32)
            pre_age = tf.reduce_sum(predict)
            age = tf.cast(age, tf.float32)
            MAE += tf.reduce_sum(tf.math.abs(grd_age - age))

            if i % 1000 == 0:
                print('{} image(s) for MAE = {}'.format(i + 1, MAE / (i + 1)))

        print('Total MAE = {}'.format(MAE / FLAGS.n_test))

if __name__ == '__main__':
     app.run(main)

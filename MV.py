#-*- coding: utf-8 -*-
# Mean-Variance Loss for Deep Age Estimation from a Face
from model import *
from absl import flags,app
from random import shuffle

import tensorflow as tf
import numpy as np
import time
import os
import sys
import datetime

flags.DEFINE_string('img_path', '/yuwhan/yuwhan/Dataset/[1]Third_dataset/Morph/All/Crop_dlib/', 'Image directory')

flags.DEFINE_string('txt_path', '/yuwhan/yuwhan/Dataset/[2]Fourth_dataset/age_banchmark/train_data/Morph/train.txt', 'Text (with label information) directory')

flags.DEFINE_string('val_img_path', '/yuwhan/yuwhan/Dataset/[1]Third_dataset/Morph/All/Crop_dlib/', 'Validate image path')

flags.DEFINE_string('val_txt_path_1', '/yuwhan/yuwhan/Dataset/[2]Fourth_dataset/age_banchmark/train_data/Morph/test.txt', 'Validate text path')

flags.DEFINE_string('val_txt_path_2', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_69_fullDB/[3]MegaAge_43_69_and_Morph_16_42/MORPH_test_16_42.txt', 'Validate text path')

flags.DEFINE_integer('val_batch_size', 15, "Validation batch size")

flags.DEFINE_integer('val_batch_size_2', 128, "Validation batch size")

flags.DEFINE_integer('img_size', 224, '')

flags.DEFINE_integer('ch', 3, '')

flags.DEFINE_float('lr', 0.001, '')

flags.DEFINE_bool('pre_checkpoint', False, '')

flags.DEFINE_string('pre_checkpoint_path', '', '')

flags.DEFINE_integer('epochs', 100, '')

flags.DEFINE_integer('num_classes', 48, '')

flags.DEFINE_integer('batch_size', 64, '')

flags.DEFINE_string('save_checkpoint', 'D:/tensorflor2.0(New_age_estimation)/checkpoint', '')

flags.DEFINE_bool('train', True, '')

flags.DEFINE_string("graphs", "D:/tensorflor2.0(New_age_estimation)/graphs/", "")

flags.DEFINE_string("output_loss_txt", "/yuwhan/Edisk/yuwhan/Edisk/4th_paper/age_banchmark/Morph/loss_MV.txt", "")

FLAGS = flags.FLAGS
FLAGS(sys.argv)
# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):      # 10 에폭당 0.1 씩 줄어듬
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)
        # step = 토탈 스탭, step_decay = 감소시킬 스탭

    def __call__(self, step):
        if (step + 1) % self._step_decay == 0:      # 여기를 좀더 손봐야한다
            #print(step)
            #print(self._step_decay)
            self.current_learning_rate = self.current_learning_rate * 0.1
            return self.current_learning_rate
        else:
            self.current_learning_rate
            return self.current_learning_rate

len_dataset = len(np.loadtxt(FLAGS.txt_path, dtype=np.float32, skiprows=0, usecols=1))
scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset // FLAGS.batch_size, 15 * len_dataset // FLAGS.batch_size)
optimizer = tf.keras.optimizers.SGD(scheduler)
age_list = np.arange(0, FLAGS.num_classes+1).astype(np.float32)

def _func(filename, label):

    image_string = tf.io.read_file(filename)
    decode_image = tf.image.decode_jpeg(image_string, channels=3)
    decode_image = tf.image.resize(decode_image, [FLAGS.img_size, FLAGS.img_size]) / 255.
    #decode_image = tf.image.per_image_standardization(decode_image)

    lab = label - 16.

    return decode_image, lab

def _test_func(image, label):

    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 255.
    #img = tf.image.per_image_standardization(img)
    
    label = (int(label) - 16)

    return img, label

@tf.function
def val_cal(images, labels, model):

    logits = run_model(model, images, False)
    logits = tf.nn.softmax(logits)
    
    pre_age = tf.reduce_sum(logits * age_list, 1, keepdims=True)
    pre_age = tf.round(pre_age)
    labels = tf.cast(labels, tf.float32)

    MAE = tf.reduce_sum(tf.math.abs(labels - pre_age))

    return MAE

# @tf.function
def run_model(model, images, training=True):
    logits = model(images, training=training)
    return logits

def mean_varince_loss(logits, Scalar_age):
    # Scalar_age [batch, ]
    # age_list  [classes, ]
    p = tf.nn.softmax(logits, 1)        # [batch, classes]
    m = tf.squeeze(tf.reduce_sum(age_list * p, 1, keepdims=True), 1)
    mean_loss = tf.reduce_mean( tf.square(m - (Scalar_age))) / 2.0
    
    # variance loss
    b = tf.square(age_list[None, :] - m[:, None])
    variance_loss = tf.reduce_sum(p * b, 1, keepdims=True)
    variance_loss = tf.reduce_mean(variance_loss)

    return 0.2*mean_loss, 0.05*variance_loss

def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        logits = run_model(model, images, True)
        mean_loss, variance_loss = mean_varince_loss(logits, labels)
        softmax_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)
        total_loss = softmax_loss + mean_loss + variance_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, mean_loss, variance_loss, softmax_loss

def main(argv=None):
    # Mean-Variance Loss for Deep Age Estimation from a Face
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
    h = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005), name="MV_fc1")(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    h = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005), name="MV_fc2")(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    y = tf.keras.layers.Dense(FLAGS.num_classes+1, name='last_layer')(h)

    train_model = tf.keras.Model(inputs=train_model.input, outputs=y)

    train_model.get_layer("MV_fc1").set_weights(pre_train_fc.get_layer("fc1").get_weights())
    train_model.get_layer("MV_fc2").set_weights(pre_train_fc.get_layer("fc2").get_weights())

    train_model.summary()

    if FLAGS.pre_checkpoint is True:

        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=train_model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')


    if FLAGS.train is True:

        data_name = np.loadtxt(FLAGS.txt_path, dtype='<U100', skiprows=0, usecols=0)
        data_name = [FLAGS.img_path + data_name_ for data_name_ in data_name]
        data_label = np.loadtxt(FLAGS.txt_path, dtype=np.float32, skiprows=0, usecols=1)

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
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        loss_f = open(FLAGS.output_loss_txt, "w")

        count = 0
        for epoch in range(FLAGS.epochs):
            A = list(zip(data_name, data_label))
            shuffle(A)
            data_name, data_label = zip(*A)
            data_name, data_label = np.array(data_name), np.array(data_label)

            data_generator = tf.data.Dataset.from_tensor_slices((data_name, data_label))
            data_generator = data_generator.shuffle(len(data_name))
            data_generator = data_generator.map(_func)
            data_generator = data_generator.batch(FLAGS.batch_size)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)
            it = iter(data_generator)

            batch_idx = len(data_label) // FLAGS.batch_size
            for step in range(batch_idx):

                batch_images, batch_labels = next(it)
    
                start_time = time.time()
                total_loss, mean_loss, var_loss, softmax_loss = train_step(train_model, batch_images, batch_labels)
                end_time = time.time()
    
                if count % 10 == 0:
                    print("Epoch: {} [{} / {}], total loss = {}, mean loss = {}, variance loss = {}, softmax loss = {} ({} steps)".format(epoch + 1, 
                                                                                                                                            step + 1,
                                                                                                                                            batch_idx,
                                                                                                                                            total_loss,
                                                                                                                                            mean_loss,
                                                                                                                                            var_loss,
                                                                                                                                            softmax_loss,
                                                                                                                                            count + 1))
                    print("Lenaring rate = {}".format(scheduler.current_learning_rate))

                with train_summary_writer.as_default():
                    tf.summary.scalar(u'total loss', total_loss, step=count)
                    tf.summary.scalar("mean loss", mean_loss, step=count)
                    tf.summary.scalar("variance loss", var_loss, step=count)
                    tf.summary.scalar("softmax loss", softmax_loss, step=count)

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
                            pre_age = val_cal(img, lab, train_model) # 이 함숙 고쳐야함
                            AE += pre_age

                        print("MAE = {} ({})".format(AE / len(val_img), test_list[j]))  # 왜 똑같이나오는가? 측정 함수 코드 문제
                        
                        loss_f.write("Epochs: {}, step = {}".format(epoch, count))
                        loss_f.write(" --> ")
                        loss_f.write(test_list[j])
                        loss_f.write(": ")
                        loss_f.write(str(AE / len(val_img)))
                        loss_f.write(", ")

                    loss_f.write("\n")
                    loss_f.flush()  # 모든 실험할 age estimation 코드에서 지금 이 부분을 추가로 넣을것

                    #model_dir = FLAGS.save_checkpoint
                    #folder_name = int(count // 100)
                    #folder_neme_str = '%s/%s_%s' % (model_dir, folder_name, "[3]MAE")
                    #if not os.path.isdir(folder_neme_str):
                    #    print("Make {} folder to save checkpoint".format(folder_name))
                    #    os.makedirs(folder_neme_str)
                    #checkpoint = tf.train.Checkpoint(train_model=train_model,optimizer=optimizer)
                    #checkpoint_dir = folder_neme_str + "/" + "MV_{}_steps.ckpt".format(count + 1)
                    #checkpoint.save(checkpoint_dir)

                    #with val_summary_writer.as_default():
                    #    tf.summary.scalar(u'[3]MAE', AE2 / len(val_data_label_2), step=count)
    
                count += 1
    else:
        Test()
    
if __name__ == '__main__':
    app.run(main)

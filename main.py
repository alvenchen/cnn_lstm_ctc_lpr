#coding=utf-8


import datetime
import os
import time
import sys
import cv2
import numpy as np
import tensorflow as tf

import lstm_ctc
import utils
import argparse

np.set_printoptions(threshold=100)

FLAGS = tf.app.flags.FLAGS


from logger_wrapper import setup_logger
logger = setup_logger('cnn_lstm', 'cnn_lstm.log')


def train(model, sess, saver, train_dir=None, val_dir=None,):

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

    print('loading train data')
    train_feeder = utils.DataIterator(data_dir=train_dir)
    print('size: {}'.format(train_feeder.size))
    num_train_samples = train_feeder.size  # 100000
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  # example: 100000/100


    val_feeder, num_val_samples, shuffle_idx_val = val_prepare(val_dir)

    print('=============================begin training=============================')
    for cur_epoch in range(FLAGS.num_epochs):
       
        # validation
        accuracy, val_cost = val_run(model, sess, val_feeder, num_val_samples, shuffle_idx_val)

        shuffle_idx = np.random.permutation(num_train_samples)
        train_cost = 0
        start_time = time.time()
        batch_time = time.time()
        train_acc = 0

        # the training part
        for cur_batch in range(num_batches_per_epoch):
            if (cur_batch + 1) % 100 == 0:
                print('batch', cur_batch, ': time', time.time() - batch_time)
                logger.info('batch {}   : time {}'.format(cur_batch, time.time() - batch_time))
                batch_time = time.time()
            indexs = [shuffle_idx[i % num_train_samples] for i in
                      xrange(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
            batch_inputs, _, batch_labels = \
                train_feeder.input_index_generate_batch(indexs)
            # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
            feed = {model.inputs: batch_inputs,
                    model.labels: batch_labels}

                
            train_ori_labels = train_feeder.the_label(indexs)
            
            train_dense_decoded, summary_str, batch_cost, step, _ = \
                sess.run([model.dense_decoded, model.merged_summay, model.cost, 
                    model.global_step, model.train_op], feed)

            acc = utils.accuracy_calculation(train_ori_labels, train_dense_decoded,
                                             ignore_value=-1, isPrint=False)
            train_acc += acc

            # calculate the cost
            train_cost += batch_cost

            train_writer.add_summary(summary_str, step)

        # save the checkpoint        
        if not os.path.isdir(FLAGS.checkpoint_dir):
            os.mkdir(FLAGS.checkpoint_dir)
        logger.info('save checkpoint at step {}'.format(step))
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)

        # train_err += the_err * FLAGS.batch_size

        train_acc = train_acc / num_batches_per_epoch
         

        #avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)

        # train_err /= num_train_samples
        now = datetime.datetime.now()
        log = "{}/{} {}:{}:{} Epoch {}/{}, " \
              "train_acc = {:.3f} accuracy = {:.3f},train_cost = {:.3f}, " \
              "val_cost = {:.3f}, time = {:.3f}"
        print_str = log.format(now.month, now.day, now.hour, now.minute, now.second,
                         cur_epoch + 1, FLAGS.num_epochs, train_acc, accuracy, train_cost,
                         val_cost, time.time() - start_time)
        print(print_str)
        logger.info(print_str)

def val_run(model, sess, val_feeder, num_val_samples, shuffle_idx_val):
    acc_batch_total = 0
    val_cost = 0
    lr = 0

    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size)  # example: 10000/100

    for j in range(num_batches_per_epoch_val):
        indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                      range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
        val_inputs, _, val_labels = \
            val_feeder.input_index_generate_batch(indexs_val)
        val_feed = {model.inputs: val_inputs,
                    model.labels: val_labels}

        dense_decoded, batch_cost, lr = \
            sess.run([model.dense_decoded, model.cost, model.lrn_rate],
                     val_feed)

        # print the decode result
        ori_labels = val_feeder.the_label(indexs_val)
        acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                                         ignore_value=-1, isPrint=True)
        acc_batch_total += acc
        val_cost += batch_cost

    accuracy = acc_batch_total / num_batches_per_epoch_val

    return accuracy, val_cost

def val_prepare(val_dir):
    print('loading validation data')
    val_feeder = utils.DataIterator(data_dir=val_dir)
    print('size: {}\n'.format(val_feeder.size))

    num_val_samples = val_feeder.size
    
    shuffle_idx_val = np.random.permutation(num_val_samples)

    return val_feeder, num_val_samples, shuffle_idx_val
    
def val(model, sess, img_path):
    val_feeder, num_val_samples, shuffle_idx_val = val_prepare(img_path)

    accuracy,_ = val_run(model, sess, val_feeder, num_val_samples, shuffle_idx_val)

    print("accuracy : {}".format(accuracy))



def infer(model, sess, img_path):
    # imgList = load_img_path('/home/yang/Downloads/FILE/ml/imgs/image_contest_level_1_validate/')
    imgList = utils.load_img_path(img_path)
    print(imgList[:5])

    total_steps = (len(imgList) - 1) / FLAGS.batch_size + 1

    decoded_expression = []
    for curr_step in range(total_steps):

        imgs_input = []
        seq_len_input = []
        for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
            
            #im = cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            im = cv2.imread(img).astype(np.float32) / 255.

            im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height))
            im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
            im = utils.prewhiten(im)

            imgs_input.append(im)                

        imgs_input = np.asarray(imgs_input)            

        feed = {model.inputs: imgs_input}
        logits, dense_decoded_code, log_prob = sess.run([model.logits, model.dense_decoded, model.log_prob], feed)

        for item in dense_decoded_code:
            expression = ''

            for i in item:
                if i == -1:
                    expression += ''
                else:
                    expression += utils.decode_maps[i]

            decoded_expression.append(expression)

    with open('./result.txt', 'wb') as f:
        for code in decoded_expression:
            f.write(code.encode('utf8'))
            f.write('\n')

    with open('./result_filename.txt', 'wb') as f:
        for curr_step in range(total_steps):
            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                basename = os.path.splitext(os.path.basename(img))[0]
                basename = basename.split('_')[0]
                f.write(basename)
                f.write('\n')


def main(_):

    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        model = lstm_ctc.LSTMOCR(FLAGS.mode)        
        model.build_graph()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())            

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)            
            if FLAGS.restore:
                #ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                print('restore from checkpoint{0}'.format(FLAGS.checkpoint_name))
                saver.restore(sess, FLAGS.checkpoint_name)                


            if FLAGS.mode == 'train':
                train(model, sess, saver, FLAGS.train_dir, FLAGS.val_dir)

            elif FLAGS.mode == 'infer':                
                infer(model, sess, FLAGS.infer_dir)

            elif FLAGS.mode == 'val':                
                val(model, sess, FLAGS.val_dir)
    


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--restore', type=bool, default=False)    
    parser.add_argument('--checkpoint_name', type=str, default='./checkpoint/ocr-model-101587')
    parser.add_argument('--train_dir', type=str, default='../plate_pro_images/train/')
    parser.add_argument('--val_dir', type=str, default='../plate_pro_images/val/')
    parser.add_argument('--infer_dir', type=str, default='../plate_pro_images/infer/')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args(argv)


    tf.app.flags.DEFINE_string('checkpoint_name', args.checkpoint_name, '')
    tf.app.flags.DEFINE_string('train_dir', args.train_dir, '')
    tf.app.flags.DEFINE_string('val_dir', args.val_dir, '')
    tf.app.flags.DEFINE_string('infer_dir', args.infer_dir, '')
    tf.app.flags.DEFINE_string('mode', args.mode, 'train, val or infer')
    tf.app.flags.DEFINE_integer('num_gpus', args.num_gpus, 'num of gpus')

    tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')


    tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')
    tf.app.flags.DEFINE_integer('image_height', 32, 'image height')
    tf.app.flags.DEFINE_integer('image_width', 112, 'image width')
    tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')
    tf.app.flags.DEFINE_integer('cnn_count', 6, 'count of cnn module to extract image features.')
    tf.app.flags.DEFINE_integer('out_channels', 128, 'output channels of last layer in CNN')
    tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
    tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')
    tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
    tf.app.flags.DEFINE_integer('batch_size', 1, 'the batch_size')
    tf.app.flags.DEFINE_integer('save_steps', 10000, 'the step to save checkpoint')
    tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
    tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
    tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
    tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
    tf.app.flags.DEFINE_integer('decay_steps', 50000, 'the lr decay_step for optimizer')
    tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')


    return args



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run()
    main(parse_arguments(sys.argv[1:]))



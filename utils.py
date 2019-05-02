#coding=utf-8

import os
import numpy as np
import tensorflow as tf
import cv2
import glob


charset = [ u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽",             # 0 - 7
            u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁",             # 8 - 15
            u"豫", u"鄂", u"湘", u"粤", u"桂", u"琼", u"川", u"贵",             # 16 - 23
            u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新",                   # 24 - 30
            u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9",     # 31 - 40
            u"A", u"B", u"C", u"D", u"E", u"F", u"G", u"H",                 # 41 - 48
            u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R",                 # 49 - 56
            u"S", u"T", u"U", u"V", u"W", u"X", u"Y", u"Z",                 # 57 - 64
            u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",                   # 65 - 72
            u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",                   # 73 - 80
            u"航",u"空"                                                      # 81 - 82
             ];

encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

num_classes = len(charset) + 1
maxPrintLen = 10

def get_plate_image(image_name):
    #im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    #im = cv2.imread(image_name).astype(np.float32) / 255.
    im = cv2.imread(image_name)
    
    im = preprocess_plate(im)

    return im

def preprocess_plate(im):
        
    # resize to same height, different width will consume time on padding
    im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height))
    im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
    im = prewhiten(im)

    return im

def load_img_path(images_path):
    file_names = glob.glob("{}/*.jpg".format(images_path))
    file_names.sort()

    file_names = np.asarray(file_names)

    return file_names

class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []

        files = glob.glob("{}/*.jpg".format(data_dir))

        for image_name in files:
            try:                
                im = get_plate_image(image_name)
                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                #code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                file_name = os.path.basename(image_name).split('.')[0]
                label = file_name.split('_')[0]
                #print(code)
                code = []                
                for l in unicode(label, 'utf-8'):
                    if l not in encode_maps:
                        print('{} not in maps -- {}'.format(l, image_name))
                        continue
                    code.append(encode_maps[l])
                #code = [encode_maps[c] for c in code]
                self.labels.append(code)                
            except Exception, e:
                print("exception:{} -- {}".format(str(e), image_name))                

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):        
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            # 64 is the output channels of the last layer of CNN
            lengths = np.asarray([FLAGS.out_channels for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y 

def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    with open('./test.csv', 'a') as f:
        for i, origin_label in enumerate(original_seq):
            decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
            if isPrint and i < maxPrintLen:
                # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))
            
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

            if origin_label == decoded_label:
                count += 1


    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)    
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt', 'wb') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs

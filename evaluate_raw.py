#coding=utf-8

import glob
import os
import sys
import time
import detect
import cv2
import json
import numpy as np
import utils
import cnn_lstm_otc_ocr
import tensorflow as tf



data_dir = '../功能车牌图像库数据/'

def get_files(data_dir):
    files_save = [] 

    dirs = os.listdir(data_dir)
    for d in dirs:
        # 只检测普通车牌，不检测功能车牌
        if d.decode('utf-8')[0] != u"“":
            continue
            pass
        full_path = os.path.join(data_dir, d)
        files = glob.glob("{}/*.jpg".format(full_path))
        files_save += files

    return files_save

def get_files_name(files):
    names_save = []

    for f in files:     
        name = os.path.splitext(os.path.basename(f))[0]
        names_save.append(name)

    return names_save



def recognize(image, plate_rec_model):    

    time1 = time.time()
    images = detect.detectPlateRough(image, image.shape[0],top_bottom_padding_rate=0.1)
    time2 = time.time()

    #print("detect time : {}".format(time2-time1))

    jsons = []    

    for j,plate in enumerate(images):
        plate, rect, origin_plate = plate

        try:
            plate = utils.preprocess_plate(plate)
        except Exception, e:
            print("preprocess_plate err:{}".format(str(e)))
            continue
        
        decoded = plate_rec_model.rec(plate)

        if len(decoded) > 0:

            res_json = {}
            # TODO confidence
            res_json["Name"] = decoded[0]
            #res_json["Confidence"] = confidence;
            res_json["x"] = int(rect[0])
            res_json["y"] = int(rect[1])
            res_json["w"] = int(rect[2])
            res_json["h"] = int(rect[3])
            jsons.append(res_json)

    return json.dumps(jsons,ensure_ascii=False,encoding="gb2312"), jsons


class PlateRecModel():
    def __init__(self, model_path):
        self.model = cnn_lstm_otc_ocr.LSTMOCR("val")
        self.model.build_graph()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
    
        self.sess.run(tf.global_variables_initializer()) 
        print('restore from checkpoint{0}'.format(model_path))
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, model_path)                

    def rec(self, img):
        imgs_input = []
        imgs_input.append(img)
        decoded_expression = []

        imgs_input = np.asarray(imgs_input)
        feed = {self.model.inputs: imgs_input}
        
        dense_decoded_code = self.sess.run(self.model.dense_decoded, feed)

        for item in dense_decoded_code:
            expression = ''

            for i in item:
                if i == -1:
                    expression += ''
                else:
                    expression += utils.decode_maps[i]

            decoded_expression.append(expression)

        return decoded_expression

def main(arg):
    files = get_files(data_dir)

    n = len(files)
    print(n)

    names = get_files_name(files)

    plate_rec_model = PlateRecModel("./checkpoint/ocr-model-101587")

    false_count = n
    not_rec = 0
    align_error = 0

    for i, f in enumerate(files):
        img = cv2.imread(f)
        label = names[i].decode('utf-8')

        time1 = time.time()
        _, res_json = recognize(img, plate_rec_model)
        time2 = time.time()

        #print("time : {}".format(time2-time1))                
        if len(res_json) == 0 :
            not_rec += 1
            continue
        elif len(res_json) > 1:
            print('file : {}  rec > 1 : {}'.format(names[i], len(res_json)))
            align_error += 1
        if res_json[0]["Name"] == label:
            false_count -= 1
        
    
    print("not rec : {}".format(not_rec))
    print("align_error : {} - {}".format(align_error, float(align_error) / n))
    print("precision : {}".format(float(n - false_count) / n))

if __name__ == '__main__':
    tf.app.run()

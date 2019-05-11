
from evaluate_raw import PlateRecModel, recognize
import time
from scipy import misc
import cv2
import numpy as np
import json
from flask import Flask, request, Response

from logger_wrapper import setup_logger
logger = setup_logger('api', 'api.log')

app = Flask(__name__)

plate_rec_model = PlateRecModel("./model/ocr-model-101587")


@app.route("/PlateRec", methods=['POST'])
def PlateRec():

    try:
        logger.info("Get Request From {}".format(request.remote_addr))

        request_img = request.files['image']        

        if request_img:
            img_str = request_img.read()
            img_np = np.fromstring(img_str, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            #t0 = time.time()
            res_json_dump, res_json = recognize(img, plate_rec_model)
            #print('time cost {}'.format(time.time() - t0))
            logger.info("Return {}".format(res_json))

            return res_json_dump
    except Exception, e:
        print("Exception : {}".format(str(e)))
        pass
    return json.dumps([])


if __name__ == '__main__':    

    app.run("0.0.0.0",port=8088)

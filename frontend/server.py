import os
from flask import Flask,jsonify,render_template, request, session, Response, redirect

from datetime import datetime
from sqlalchemy import and_
import json
import threading
import time
import pandas as pd
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import backend.busquedas
import face_recognition


key_users = 'users'
cache = {}
lock = threading.Lock()
app = Flask(__name__, template_folder= "static/html")

# app.config["UPLOAD_FOLDER"] = "upload/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



@app.route('/')
def index():
    return render_template("mainimage.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    # print(request.__dict__)
    if 'file' not in request.files:
        message = {'msg': 'No file part!'}
        json_msg = json.dumps(message)
        return Response(json_msg, status=401, mimetype="application/json")
    
    file = request.files['file']
    typesearch=request.form.get("typesearch")

    if file.filename == '':
        message = {'msg': 'No image selected for uploading'}
        json_msg = json.dumps(message)
        return Response(json_msg, status=401, mimetype="application/json")

    if file and allowed_file(file.filename):
        imagesave=file.filename+typesearch
        print(imagesave)
        if imagesave in cache and (datetime.now()-cache[imagesave]['datetime']).total_seconds()<1800:
            output = cache[imagesave]['data']
            print(output)
            print("using cache")
        else:
            print("using matlab")
            output = search (file,typesearch,imagesave)
            print(output)
        return Response(output, status=201, mimetype="application/json")
    else:
        message = {'msg': 'Allowed image types are -> png, jpg, jpeg, gif'}
        json_msg = json.dumps(message)
        return Response(json_msg, status=401, mimetype="application/json")



def search(file,typeserach,imagesave):
    
    image = Image.open(file)

    if(int(typeserach)==1):#knn_seq
        ldImage = backend.busquedas.knnSequential()
        
    elif(int(typeserach)==2):#knn_rtree
        ldImage = backend.busquedas.knnRtree()
    else:#range_search
        answer=[]

    if(len(answer)>0):
        np_x = np.array(answer[0]._data).reshape(answer[0].size, order='F')
        output=pd.Series(np_x).to_json(orient='values')
        now = datetime.now()
        cache[imagesave] = {'data':output, 'datetime':now}
        return output

    return []



if __name__ == '__main__':
    app.secret_key = ".."
    # app.run(port=4224, threaded=True, host=('172.31.74.220'))
    app.run(port=8080, threaded=True, host=('127.0.0.1'))
    #app.run(port=1111, threaded=True, host=('3.138.193.36'))
    #app.run(port=80, threaded=True, host=('0.0.0.0'))
    #app.run(port=80, threaded=True, host=('0.0.0.0/0'))
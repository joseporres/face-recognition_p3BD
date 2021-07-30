import os
from flask import Flask,jsonify,render_template, request, session, Response, redirect

from datetime import datetime
from sqlalchemy import and_
import json
import threading
import time
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename



from busquedas import knnSequential
from busquedas import knnRtree
from time import time

import face_recognition


key_users = 'users'
cache = {}
lock = threading.Lock()
app = Flask(__name__, template_folder= "static/html")

app.config["UPLOAD_FOLDER"] = "fotos/"
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
    k = request.form.get("k")
    #print(k)
    n = request.form.get("n")
    #print(n)

    if file.filename == '':
        message = {'msg': 'No image selected for uploading'}
        json_msg = json.dumps(message)
        return Response(json_msg, status=401, mimetype="application/json")

    if file and allowed_file(file.filename):
        
        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        output = search(filename,typesearch,int(k),int(n))

        return Response(json.dumps(output), status=201, mimetype="application/json")
    else:
        message = {'msg': 'Allowed image types are -> png, jpg, jpeg, gif'}
        json_msg = json.dumps(message)
        return Response(json_msg, status=401, mimetype="application/json")



def search(filename,typeserach,k,n):

    if(int(typeserach)==1):#knn_seq
        #print("Knn seq")
        path = 'fotos/' + filename
        #print(path)
        Q = face_recognition.face_encodings(face_recognition.load_image_file(path))[0]
        start_time = time()
        ldImage = knnSequential(k,Q,n)
        #print(time() - start_time)

        #print(ldImage)
        
        return ldImage
        
             
    elif(int(typeserach)==2):#knn_rtree
        #print("Knn rtree")
        path = 'fotos/' + filename
        #print(path)
        Q = face_recognition.face_encodings(face_recognition.load_image_file(path))[0]

        start_time = time()
        ldImage = knnRtree(k,Q,n)
        #print(time() - start_time)
        # print(ldImage)
        return ldImage
        

    return {}

    # if(len(answer)>0):
    #     np_x = np.array(answer[0]._data).reshape(answer[0].size, order='F')
    #     output=pd.Series(np_x).to_json(orient='values')
    #     now = datetime.now()
    #     cache[imagesave] = {'data':output, 'datetime':now}
    #     return output

    



if __name__ == '__main__':
    app.secret_key = ".."
    # app.run(port=4224, threaded=True, host=('172.31.74.220'))
    app.run(port=8080, threaded=True, host=('127.0.0.1'))
    #app.run(port=1111, threaded=True, host=('3.138.193.36'))
    #app.run(port=80, threaded=True, host=('0.0.0.0'))
    #app.run(port=80, threaded=True, host=('0.0.0.0/0'))
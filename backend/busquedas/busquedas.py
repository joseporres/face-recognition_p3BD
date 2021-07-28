import face_recognition
from rtree import index 
import os
import heapq

def knnSequential(k, Q, n, path):  
    conocidas = []
    names_in_order = []
    break_fg = False
    c = 0  

    for file_path in os.listdir(path):
        auxPath = path + "/" + file_path

        imgList = os.listdir(auxPath)

        for filename in imgList: 
            auxPath2 = auxPath + "/" + filename

            img = face_recognition.load_image_file(auxPath2)

            unknown_face_encodings = face_recognition.face_encodings(img)

            for elem in unknown_face_encodings:
                if c == n: #Process n_images 
                    distances = face_recognition.face_distance(conocidas, Q)
                    auxArr = []
                    for i in range(0, c, 1):
                        auxArr.append((distances[i], names_in_order[i]))
                    heapq.heapify(auxArr)    
                    return heapq.nsmallest(k, auxArr)

                names_in_order.append(auxPath2)
                conocidas.append(elem)        
                c = c + 1


    distances = face_recognition.face_distance(conocidas, Q)
    auxArr = [] 

    for i in range(0, c, 1):
        auxArr.append((distances[i], names_in_order[i]))
    heapq.heapify(auxArr)    
    return heapq.nsmallest(k, auxArr)


def knnRtree(k, Q):

  p = index.Property()
  p.dimension = 128 
  p.buffering_capacity = 10 
 
  Rtree = index.Rtree("../preprocess/Rtree", properties=p)  
  listQ = list(Q)
  for i in Q:
    listQ.append(i)

  
  return list(Rtree.nearest(coordinates=listQ, num_results=k, objects='raw'))

print(knnRtree(5
, face_recognition.face_encodings(face_recognition.load_image_file('../fotos_prueba/unknown.jpg'))[0]))

# print(knnSequential(5
# , face_recognition.face_encodings(face_recognition.load_image_file('../fotos_prueba/unknown.jpg'))[0]
# , 100
# , '../preprocess/lfw'))


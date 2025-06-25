import cv2
import matplotlib.pyplot as plt

# video=cv2.VideoCapture(0)
# while True:
#     ret,frame=video.read()
#     cv2.imshow("Age-Gender",frame)
#     k=cv2.waitKey(1)
#     if k==ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()


image = cv2.imread("0Q6A9766.JPG")
image = cv2.resize(image,(600,600))

face_pbtxt="models/opencv_face_detector.pbtxt"
face_pb="models/opencv_face_detector_uint8.pb"
age_model="models/age_net.caffemodel"
age_prototxt="models/age_deploy.prototxt"
gender_prototxt="models/gender_deploy.prototxt"
gender_model="models/gender_net.caffemodel"

#Load Models
face=cv2.dnn.readNet(face_pb,face_pbtxt)
age=cv2.dnn.readNet(age_model,age_prototxt)
gender=cv2.dnn.readNet(gender_model,gender_prototxt)

#Setup Classifications
age_classification=['(0-2)','(3-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
gender_classification=['Male','Female']

#Image Copy
img_cp=image.copy()

img_height=img_cp.shape[0]
img_width=img_cp.shape[1]
blob = cv2.dnn.blobFromImage(img_cp,1.0,(200,200),[104,117,123],True,False)

face.setInput(blob)
detected_faces=face.forward()
# print(f"Confidence: {detected_faces[0,0]}")
face_bounds=[]

for i in range(detected_faces.shape[2]):
    confidence=detected_faces[0,0,i,2]
    if (confidence>0.7):
        print(detected_faces[0,0,i],)
        x1=int(detected_faces[0,0,i,3]*img_width)
        y1=int(detected_faces[0,0,i,4]*img_height)
        x2=int(detected_faces[0,0,i,5]*img_width)
        y2=int(detected_faces[0,0,i,6]*img_height)
        # print([x1,y1,x2,y2])
        cv2.rectangle(img_cp,(x1,y1),(x2,y2),(0,0,255),int(round(img_height/150)),8)
        face_bounds.append([x1,y1,x2,y2])

for face_bound in face_bounds:
    try:
        face=img_cp[max(0,face_bound[1]-15): min(face_bound[3]+15,img_cp.shape[0]-1),
                    max(0,face_bound[0]-15): min(face_bound[2]+15,img_cp.shape[1]-1)]
    
        blob= cv2.dnn.blobFromImage(face, 1.0, (277,277),[104,117,123],True)
        gender.setInput(blob)
        gender_prediction=gender.forward()
        print(gender_prediction)
    except Exception as e:
        print(e)

# plt.imshow(img_cp)
# plt.axis('off')  # Hide axis
# plt.show()

while True:
    cv2.imshow("result",img_cp)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
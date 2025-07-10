import cv2
import matplotlib.pyplot as plt

image = cv2.imread("dataset.png")
image = cv2.resize(image,(720,640))

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

img_cp=image.copy()

def bounding_box(img_cp):

    img_height=img_cp.shape[0]
    img_width=img_cp.shape[1]
    blob = cv2.dnn.blobFromImage(img_cp,1.0,(300,300),[104,117,123],True,False)


    face.setInput(blob)
    detected_faces=face.forward()
    # print(f"Confidence: {detected_faces[0,0]}")
    face_bounds=[]

    print("Blob shape:", blob.shape)
    print("Detected faces shape:", detected_faces.shape)

    for i in range(detected_faces.shape[2]):
        confidence=detected_faces[0,0,i,2]
        if (confidence>0.9):
            print(detected_faces[0,0,i],)
            x1=int(detected_faces[0,0,i,3]*img_width)
            y1=int(detected_faces[0,0,i,4]*img_height)
            x2=int(detected_faces[0,0,i,5]*img_width)
            y2=int(detected_faces[0,0,i,6]*img_height)
            # print([x1,y1,x2,y2])
            cv2.rectangle(img_cp,(x1,y1),(x2,y2),(0,0,255),2)
            print(f"Confidence: {confidence:.2f} | Face at: ({x1}, {y1}) to ({x2}, {y2})")
            face_bounds.append([x1,y1,x2,y2])

    return face_bounds

#Setup Classifications
age_classification=['(0-2)','(3-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
gender_classification=['Male','Female']

im_age=image.copy()

#Image Copy

def age_gender_prediction(im_age,face_bounds):
    for face_bound in face_bounds:
        try:
            _face=im_age[max(0,face_bound[1]-25): min(face_bound[3]+25,img_cp.shape[0]-1),
                        max(0,face_bound[0]-20): min(face_bound[2]+20,img_cp.shape[1]-1)]
            

            _face_rgb = cv2.cvtColor(_face, cv2.COLOR_BGR2RGB)

            # Display using matplotlib
            plt.imshow(_face_rgb)
            plt.title("Cropped Face")
            plt.axis('off')
            plt.show()
        
            _blob= cv2.dnn.blobFromImage(_face, 1.0, (227,227),[78.4263377603, 87.7689143744, 114.895847746],)
            gender.setInput(_blob)
            predicted_gender=gender.forward()
            # print("predicted_gender",predicted_gender)
            _gender= gender_classification[predicted_gender[0].argmax()]
            print("Gender :",_gender)

            age.setInput(_blob)
            predicted_age=age.forward()
            # print("predicted_age",predicted_age)
            _age=age_classification[predicted_age[0].argmax()]
            print("AGE :",_age)

            cv2.putText(img_cp, f"{_gender}, {_age}", (face_bound[0], face_bound[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img_cp
        except Exception as e:
            print(e)


while True:
    face_box=bounding_box(img_cp)
    prediction_image=age_gender_prediction(img_cp,face_box)
    cv2.imshow("result",prediction_image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
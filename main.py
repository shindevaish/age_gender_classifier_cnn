import cv2
import matplotlib.pyplot as plt

def load_models():
    face = cv2.dnn.readNet("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")
    age = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
    gender = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")
    return face, age, gender

def bounding_box(img_cp, face):
    img_height = img_cp.shape[0]
    img_width = img_cp.shape[1]
    blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), [104, 117, 123], True, False)

    face.setInput(blob)
    detected_faces = face.forward()
    face_bounds = []

    print("Blob shape:", blob.shape)
    print("Detected faces shape:", detected_faces.shape)

    for i in range(detected_faces.shape[2]):
        confidence = detected_faces[0, 0, i, 2]
        if confidence > 0.9:
            x1 = int(detected_faces[0, 0, i, 3] * img_width)
            y1 = int(detected_faces[0, 0, i, 4] * img_height)
            x2 = int(detected_faces[0, 0, i, 5] * img_width)
            y2 = int(detected_faces[0, 0, i, 6] * img_height)
            cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print(f"Confidence: {confidence:.2f} | Face at: ({x1}, {y1}) to ({x2}, {y2})")
            face_bounds.append([x1, y1, x2, y2])

    return face_bounds

def age_gender_prediction(img_cp, im_age, face_bounds, age, gender):
    age_classification = ['(0-2)', '(3-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_classification = ['Male', 'Female']

    for face_bound in face_bounds:
        try:
            _face = im_age[max(0, face_bound[1]-25): min(face_bound[3]+25, im_age.shape[0]-1),
                           max(0, face_bound[0]-20): min(face_bound[2]+20, im_age.shape[1]-1)]


            _blob = cv2.dnn.blobFromImage(_face, 1.0, (227, 227),
                                          [78.4263377603, 87.7689143744, 114.895847746])

            gender.setInput(_blob)
            predicted_gender = gender.forward()
            _gender = gender_classification[predicted_gender[0].argmax()]
            print("Gender :", _gender)

            age.setInput(_blob)
            predicted_age = age.forward()
            _age = age_classification[predicted_age[0].argmax()]
            print("AGE :", _age)

            cv2.putText(img_cp, f"{_gender}, {_age}", (face_bound[0], face_bound[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except Exception as e:
            print("Error in face processing:", e)

    return img_cp

def main():
    image = cv2.imread("dataset.png")
    image = cv2.resize(image, (720, 640))

    face, age, gender = load_models()
    img_cp = image.copy()
    im_age = image.copy()

    face_box = bounding_box(img_cp, face)
    prediction_image = age_gender_prediction(img_cp, im_age, face_box, age, gender)

    cv2.imshow("result", prediction_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

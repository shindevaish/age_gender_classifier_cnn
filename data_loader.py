import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Load OpenCV Face Detection Model
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"

face_net = cv2.dnn.readNet(face_pb, face_pbtxt)


class FaceDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.image_files)

    def detect_face(self, image):

        image_np = np.array(image)

        h, w = image_np.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image_np,
            1.0,
            (300, 300),
            [104, 117, 123],
            swapRB=False,
            crop=False
        )

        face_net.setInput(blob)

        detections = face_net.forward()

        best_face = None
        best_confidence = 0

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5 and confidence > best_confidence:

                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                best_face = (x1, y1, x2, y2)
                best_confidence = confidence

        if best_face is not None:

            x1, y1, x2, y2 = best_face

            # Add small padding
            padding = 20

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            face = image_np[y1:y2, x1:x2]

            if face.size > 0:
                return Image.fromarray(face)

        # Return original image if no face detected
        return image

    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        try:
            parts = img_name.split('_')

            age = int(parts[0])
            gender = int(parts[1])

        except (IndexError, ValueError):

            print(f"ERROR parsing filename: {img_name}")
            return None, None

        image = Image.open(img_path).convert("RGB")

        # Detect and crop face
        image = self.detect_face(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(
            [age, gender],
            dtype=torch.float32
        )


def get_transforms():

    return transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


dir = "augmented_dataset"

dataset_ = FaceDataset(
    root_dir=dir,
    transform=get_transforms()
)

dataloader = DataLoader(
    dataset_,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
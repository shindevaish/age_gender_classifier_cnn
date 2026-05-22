import sys
import os

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(ROOT_DIR)

import torch

from model import AgeGenderModel
from utils.transform import transform

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = AgeGenderModel().to(device)

model.load_state_dict(
    torch.load("age_gender_augmented_model.pth", map_location=device)
)

model.eval()


def predict(image):

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        age_output, gender_output = model(image)

        predicted_age = int(age_output.item())

        predicted_gender = torch.argmax(
            gender_output,
            dim=1
        ).item()

        gender = "Male" if predicted_gender == 0 else "Female"

    return predicted_age, gender
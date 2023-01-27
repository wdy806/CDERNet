import os
from PIL import Image, ImageDraw
import face_recognition

pic_path = 'datasets/DEFE/raw_data/'
target_path = 'datasets/DEFE/cropped_data/'

pic_list = os.listdir(pic_path)
for pic in pic_list:
    image = face_recognition.load_image_file(pic_path + pic)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
    face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
    for face_location in face_locations:
        top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    print(pic)
    
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for face_landmarks in face_landmarks_list:
        facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
        ]
        for facial_feature in facial_features:
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], width=2, fill=(255, 0, 0))
    pil_image.show()
    
    pil_image = Image.fromarray(face_image)
    pil_image = pil_image.resize((224, 224), 3)
    pil_image.save(target_path + pic)
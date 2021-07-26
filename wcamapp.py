from PIL import Image, ImageDraw
import face_recognition
import os
import cv2
from playsound import playsound
import keras

class CamApp():
    def __init__(self, model):
        self.model = keras.models.load_model(model)

    def eye_cropper(self, image):
        count = 0
        face_landmarks_list = face_recognition.face_landmarks(image)

        eyes = []
        try:
            eyes.append(face_landmarks_list[0]['left_eye'])
            eyes.append(face_landmarks_list[0]['right_eye'])
        except:
            pass

        for eye in eyes:
            x_max = max([coordinate[0] for coordinate in eye])
            x_min = min([coordinate[0] for coordinate in eye])
            y_max = max([coordinate[1] for coordinate in eye])
            y_min = min([coordinate[1] for coordinate in eye])
            x_range = x_max - x_min
            y_range = y_max - y_min

            if x_range > y_range:
                right = round(.5*x_range) + x_max
                left = x_min - round(.5*x_range)
                bottom = round(((right-left) - y_range))/2 + y_max
                top = y_min - round(((right-left) - y_range))/2
            else:
                bottom = round(.5*y_range) + y_max
                top = y_min - round(.5*y_range)
                right = round(((bottom-top) - x_range))/2 + x_max
                left = x_min - round(((bottom-top) - x_range))/2

                im = Image.open(image)
                im = im.crop((left, top, right, bottom))

                im = im.resize((80,80))

                count += 1
                if count % 200 == 0:
                    print(count)
                
                return im
    
    def start_app(self):
        cap = cv2.VideoCapture(0)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if not cap.isOpened():
            raise IOError('Cannot open webcam')

        while True:
            ret, frame = cap.read()
            image_for_prediction = self.eye_cropper(frame)
            try:
                image_for_prediction = image_for_prediction/255.0
            except:
                continue
        
            prediction = self.model.predict(image_for_prediction)

            if prediction < 0.5:
                counter = 0
                status = 'Open'
                cv2.putText(frame, status, (round(w/2)-80,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_4)
            else:
                counter = counter + 1
                status = 'Closed'
                cv2.putText(frame, status, (round(w/2)-104,70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_4)

            if counter > 5:
                cv2.putText(frame, 'SLEEPING', 
                        (round(w/2)-136,round(h) - 146), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_4)
                counter = 5

            cv2.imshow('Sleeping Detection', frame)
            k = cv2.waitKey(1)
            if k == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()


        
app = CamApp('18_96_650fotos_sigmoid.h5')
app.start_app()




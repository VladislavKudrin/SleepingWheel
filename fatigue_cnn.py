
import cv2
import dlib
from tensorflow import keras
eye_model = keras.models.load_model('./models/eyes/300_15/23_95_sigmoid_normal.h5')
mouth_model = keras.models.load_model('./models/mouth/300_15/1_94_sigmoid_normal.h5')
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# frame into the predictor
def cropper(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    dets = detector(img, 1)
    for d in (dets):
        shape = predictor(img, d)
        xlefteyes = [shape.part(x).x for x in range(36,41)]
        ylefteyes = [shape.part(x).y for x in range(36,41)]
        le_maxx = max(xlefteyes)
        le_minx = min(xlefteyes)
        le_maxy = max(ylefteyes)
        le_miny = min(ylefteyes)

        xrighteyes = [shape.part(x).x for x in range(42,47)]
        yrighteyes = [shape.part(x).y for x in range(42,47)]
        re_maxx = max(xrighteyes)
        re_minx = min(xrighteyes)
        re_maxy = max(yrighteyes)
        re_miny = min(yrighteyes)

        xmouthpoints = [shape.part(x).x for x in range(48,67)]
        ymouthpoints = [shape.part(x).y for x in range(48,67)]
        m_maxx = max(xmouthpoints)
        m_minx = min(xmouthpoints)
        m_maxy = max(ymouthpoints)
        m_miny = min(ymouthpoints) 

        # to show the mouth properly pad both sides
        pad = 10
        image_for_prediction = []
        crop_image_le = frame[le_miny-pad:le_maxy+pad,le_minx-pad:le_maxx+pad]
        cv2.rectangle(frame, (le_minx-pad, le_miny-pad), (le_maxx+pad, le_maxy+pad), (255,0,0), 2)
        image_for_prediction.append(crop_image_le)

        crop_image_re = frame[re_miny-pad:re_maxy+pad,re_minx-pad:re_maxx+pad]
        cv2.rectangle(frame, (re_minx-pad,re_miny-pad), (re_maxx+pad, re_maxy+pad), (0,0,255), 2)
        image_for_prediction.append(crop_image_re)

        crop_image = frame[m_miny-pad:m_maxy+pad,m_minx-pad:m_maxx+pad]
        cv2.rectangle(frame, (m_minx-pad,m_miny-pad), (m_maxx+pad, m_maxy+pad), (0,255,0), 2)
        image_for_prediction.append(crop_image)

        
        for i, img in enumerate(image_for_prediction):
            img = cv2.resize(img, (80,80))
            img = img.reshape(-1, 80, 80, 3)
            image_for_prediction[i] = img


        return image_for_prediction

# initiate webcam
cap = cv2.VideoCapture(0)

EYE_THRESH = 0.70
MOUTH_THRESH = 0.3


# create a while loop that runs while webcam is in use
while True:
    ret, frame = cap.read()
    image_for_prediction = cropper(frame)
    try:
        left_eyes = image_for_prediction[0]
        right_eyes = image_for_prediction[1]
        mouth = image_for_prediction[2]
        left_eyes = left_eyes/255.0
        right_eyes = right_eyes/255.0
        mouth = mouth/255.0
    except:
        continue

    prediction_le = eye_model.predict(left_eyes)
    prediction_re = eye_model.predict(right_eyes)
    prediction_mouth = mouth_model.predict(mouth)

    prediction_eyes = (prediction_le + prediction_re) / 2

    print(f"left_eyes={prediction_le[0]} right_eyes={prediction_re[0]} mouth={prediction_mouth[0]}")

    if (prediction_eyes > EYE_THRESH and prediction_mouth > MOUTH_THRESH):
        cv2.putText(frame, "Yawn", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if (prediction_mouth > MOUTH_THRESH):
        cv2.putText(frame, "Mouth open", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if (prediction_eyes > EYE_THRESH):
        cv2.putText(frame, "Eyes closed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Fatigue Detector', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
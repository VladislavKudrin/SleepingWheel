import sys
import os
import dlib
import glob
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = "./origin_dataset/test/no_yawn/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = cv2.imread(f)
    img_rgb = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        xkeypoints = [shape.part(x).x for x in range(36,41)]
        ykeypoints = [shape.part(x).y for x in range(36,41)]
        maxx = max(xkeypoints)
        minx = min(xkeypoints)
        maxy = max(ykeypoints)
        miny = min(ykeypoints) 
        pad = 10
        filename = os.path.splitext(os.path.basename(f))[0]

        crop_image = img[miny-pad:maxy+pad,minx-pad:maxx+pad]
        crop_rgb = img_rgb[miny-pad:maxy+pad,minx-pad:maxx+pad]
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2RGB) 
        cv2.imwrite(filename+'.jpg',crop_rgb)
    
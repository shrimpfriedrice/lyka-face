import os
import shlex
import subprocess
#http://opencv.org/
import cv2
#https://github.com/cmusatyalab/openface
import openface
import openface.helper
from openface.data import iterImgs

base_path = os.path.dirname(os.path.realpath(__file__))
raw_faces_path = os.path.join(base_path, 'images/raw')
aligned_faces_path = os.path.join(base_path, 'images/aligned')
dlib_face_predictor = os.path.join(base_path, 'models/dlib/shape_predictor_68_face_landmarks.dat')
generate_representations_cache = os.path.join(base_path, 'images/aligned/cache.t7')

aligned_image_size = 96
align_images_with_multiple_faces = False
align_landmark_indices = openface.AlignDlib.OUTER_EYES_AND_NOSE #or openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP


def align_faces_dlib():
    openface.helper.mkdirP(aligned_faces_path)

    images = list(iterImgs(raw_faces_path))

    align_dlib = openface.AlignDlib(dlib_face_predictor)

    for image in images:
        aligned_person_face_path = os.path.join(aligned_faces_path, image.cls)
        openface.helper.mkdirP(aligned_person_face_path)
        aligned_numbered_person_face_path = os.path.join(aligned_person_face_path, image.name)
        aligned_image_name = aligned_numbered_person_face_path + ".png"

        image_rgb = image.getRGB()
        if image_rgb is None:
            print("Unable to load image {}").format(image.name)
            aligned_image_rgb = None
        else:
            aligned_image_rgb = align_dlib.align(aligned_image_size, image_rgb, landmarkIndices=align_landmark_indices, skipMulti = align_images_with_multiple_faces)
            if aligned_image_rgb is None:
                print("Unable to align image {}").format(image.name)

        if aligned_image_rgb is not None:
            aligned_image_bgr = cv2.cvtColor(aligned_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(aligned_image_name, aligned_image_bgr)


def generate_representations_openface():
    #if your dataset has changed, delete the cache file
    if os.path.isfile(generate_representations_cache):
        print ('cache exists, im going to remove')
        os.remove(generate_representations_cache)

    generate_representations_lua = subprocess.Popen(shlex.split('./batch-represent/main.lua -outDir features -data images/aligned'))
    generate_representations_lua.wait()


align_faces_dlib()
generate_representations_openface()



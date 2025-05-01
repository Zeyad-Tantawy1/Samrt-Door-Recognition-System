# ======================
# ✅ SETUP + IMPORTS
# ======================
import imghdr
import os
import cv2
import torch
import dlib
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from datetime import datetime
from Augmentation import create_augmented_dataset
import requests


LANDMARK_PATH = "C:/Users/zeyad/PyCharmMiscProject/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_PATH)


def preprocess_image(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)




def show_image(img, title="Result"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()



def diagnose_and_fix_images(known_faces_dir, output_dir=None):

    if output_dir is None:
        output_dir = known_faces_dir + "_fixed"

    os.makedirs(output_dir, exist_ok=True)

    # Initialize detectors
    detector = dlib.get_frontal_face_detector()

    print("Starting image diagnosis...")

    for person_name in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        # Create output folder for this person
        output_person_folder = os.path.join(output_dir, person_name)
        os.makedirs(output_person_folder, exist_ok=True)

        for filename in os.listdir(person_folder):
            file_path = os.path.join(person_folder, filename)

            if not os.path.isfile(file_path) or imghdr.what(file_path) not in ['jpeg', 'jpg', 'png', 'bmp']:
                continue

            print(f"Diagnosing {file_path}...")

            try:
                # Load image
                img = cv2.imread(file_path)
                if img is None:
                    print(f"⚠️ Cannot read image: {file_path}")
                    continue

                # Diagnose image quality
                is_too_dark = cv2.mean(img)[0] < 40
                is_too_bright = cv2.mean(img)[0] > 220
                is_low_contrast = np.std(img) < 30
                is_too_small = img.shape[0] < 100 or img.shape[1] < 100

                # Apply fixes based on diagnosis
                fixed_img = img.copy()

                if is_too_dark:
                    print(f"Fixing dark image: {file_path}")
                    # Increase brightness
                    alpha = 1.5  # Contrast control
                    beta = 30    # Brightness control
                    fixed_img = cv2.convertScaleAbs(fixed_img, alpha=alpha, beta=beta)

                if is_too_bright:
                    print(f"Fixing bright image: {file_path}")
                    # Decrease brightness
                    alpha = 1.0  # Contrast control
                    beta = -30   # Brightness control
                    fixed_img = cv2.convertScaleAbs(fixed_img, alpha=alpha, beta=beta)

                if is_low_contrast:
                    print(f"Fixing low contrast image: {file_path}")
                    # Enhance contrast with CLAHE
                    lab = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    enhanced_lab = cv2.merge((cl, a, b))
                    fixed_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                if is_too_small:
                    print(f"Fixing small image: {file_path}")
                    # Upscale image
                    scale_factor = max(1, 200 / min(fixed_img.shape[0], fixed_img.shape[1]))
                    fixed_img = cv2.resize(fixed_img, (0, 0), fx=scale_factor, fy=scale_factor)

                # Apply noise reduction
                fixed_img = cv2.fastNlMeansDenoisingColored(fixed_img, None, 10, 10, 7, 21)


                fixed_img_rgb = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(fixed_img_rgb)

                if not face_locations:
                    dlib_faces = detector(fixed_img_rgb, 1)
                    if dlib_faces:
                        face = dlib_faces[0]
                        face_locations = [(face.top(), face.right(), face.bottom(), face.left())]

                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    margin = int((bottom - top) * 0.2)  # 20% margin
                    top = max(0, top - margin)
                    bottom = min(fixed_img.shape[0], bottom + margin)
                    left = max(0, left - margin)
                    right = min(fixed_img.shape[1], right + margin)

                    face_crop = fixed_img[top:bottom, left:right]

                    output_path = os.path.join(output_person_folder, "fixed_" + filename)
                    cv2.imwrite(output_path, face_crop)
                    print(f"✅ Fixed image saved to {output_path}")

                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(face_crop_rgb)
                    if not encodings:
                        print(f"⚠️ Still unable to encode face after fixes: {file_path}")
                else:
                    print(f"⚠️ No faces detected even after fixes: {file_path}")

                    output_path = os.path.join(output_person_folder, "whole_" + filename)
                    cv2.imwrite(output_path, fixed_img)
                    print(f"ℹ️ Saved entire fixed image to {output_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    print("Diagnosis and fixes completed!")





try:
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
        model = torch.hub.load('C:/Users/zeyad/PyCharmMiscProject/yolov5-face', 'custom',
                               path_or_model='C:/Users/zeyad/PyCharmMiscProject/yolov5-face/weights/yolov5m-face.pt',
                               source='local').to('cpu')
    print("✅ YOLOv5 model loaded.")
except Exception as e:
    print(f"❌ Error loading YOLOv5 model: {e}")

def detect_faces_with_yolo(img):
    return model(img).xyxy[0]




def align_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        landmarks = predictor(gray, rect)
        left, right = (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y)
        angle = np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))
        center = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return img



def load_known_faces(known_dir, aug_dir="augmented_faces"):
    create_augmented_dataset(known_dir, aug_dir)  # Generate augmented images
    encodings, names = [], []

    for folder in [known_dir, aug_dir]:
        for person in os.listdir(folder):
            person_path = os.path.join(folder, person)
            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"⚠️ Cannot load image: {img_path}")
                    continue

                print(f"Preprocessing image: {img_name}")
                img = preprocess_image(img)

                if img is None:
                    print(f"⚠️ Preprocessing failed for {img_name}")
                    continue

                print(f"Aligning face: {img_name}")
                aligned = align_face(img)

                if aligned is None:
                    print(f"⚠️ Alignment failed for {img_name}, skipping.")
                    continue


                print(f"Detecting faces in aligned image: {img_name}")
                rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb)

                if not locs:
                    print(f"⚠️ No faces detected in {img_name}")
                    continue

                top, right, bottom, left = locs[0]
                crop = cv2.resize(aligned[top:bottom, left:right], (160, 160))

                enc = face_recognition.face_encodings(crop)
                if enc:
                    encodings.append(enc[0])
                    names.append(person)
                else:
                    print(f"⚠️ Failed to encode face for {img_name}")

    print(f"✅ Loaded {len(encodings)} known face encodings.")
    return encodings, names


def diagnose_and_fix_images(input_dir, output_dir=None):
    output_dir = output_dir or input_dir + "_fixed"
    os.makedirs(output_dir, exist_ok=True)

    for person in os.listdir(input_dir):
        in_path = os.path.join(input_dir, person)
        out_path = os.path.join(output_dir, person)
        if not os.path.isdir(in_path): continue
        os.makedirs(out_path, exist_ok=True)

        for img_name in os.listdir(in_path):
            img_path = os.path.join(in_path, img_name)
            img = cv2.imread(img_path)
            fixed = preprocess_image(img)
            if fixed is not None:
                cv2.imwrite(os.path.join(out_path, img_name), fixed)
    print("✅ Image diagnosis and fixing complete.")


def smart_door_recognition_system(frame, known_encodings, known_names, threshold=0.93):
    results = detect_faces_with_yolo(frame)
    margin = 30

    for det in results:
        x1, y1, x2, y2, conf, _ = map(int, det[:6])
        if conf < 0.7:
            continue

        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(frame.shape[1], x2 + margin), min(frame.shape[0], y2 + margin)
        face_crop = frame[y1:y2, x1:x2]

        enc = face_recognition.face_encodings(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        if not enc:
            continue

        face_enc = enc[0]
        face_norm = face_enc / np.linalg.norm(face_enc)

        best_score = -1
        best_name = "Unknown"

        for known_enc, name in zip(known_encodings, known_names):
            known_norm = known_enc / np.linalg.norm(known_enc)
            cosine_sim = np.dot(face_norm, known_norm)

            if cosine_sim > best_score:
                best_score = cosine_sim
                best_name = name

        if best_score >= threshold:
            label = f"{best_name} ({best_score:.2f})"
            color = (0, 255, 0)
            log_access(best_name, best_score)
        else:
            label = f"Unknown ({best_score:.2f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame


def log_access(name, confidence=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("access_log.txt", "a") as f:
        f.write(f"{timestamp} - {name} accessed the door (Confidence: {confidence:.2f})\n")
    print(f"✅ {name} access logged.")


def run_smart_door_camera(known_dir, aug_dir="augmented_faces"):

    fixed_dir = known_dir + "_fixed"
    print("Diagnosing and fixing problematic images...")
    diagnose_and_fix_images(known_dir, fixed_dir)

    print("Loading known faces from the fixed directory...")
    encodings, names = load_known_faces(fixed_dir, aug_dir)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        output = smart_door_recognition_system(frame, encodings, names)
        cv2.imshow("Smart Door", output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


# ======================
# ✅ MAIN CALL
# ======================
run_smart_door_camera("C:/Users/zeyad/PyCharmMiscProject/known_faces",
                      "C:/Users/zeyad/PyCharmMiscProject/augmented_faces")



def send_telegram_alert(image_path, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(image_path, 'rb') as photo:
        data = {"chat_id": chat_id}
        files = {"photo": photo}
        response = requests.post(url, data=data, files=files)
    return response.ok

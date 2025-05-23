import firebase_admin
from firebase_admin import credentials, db
import base64
import numpy as np
import face_recognition
import cv2
import io
from PIL import Image, ImageDraw, ImageFont
import os
import arabic_reshaper
from bidi.algorithm import get_display

def draw_arabic_text(frame, text, x, y, color=(0, 255, 0)):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    font = ImageFont.truetype(font_path, 28)

    draw.text((x, y), bidi_text, font=font, fill=color)
    return np.array(img_pil)

# Firebase setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://dcms-aaup-6e1e4-default-rtdb.firebaseio.com/'
})

ref = db.reference('users')
users = ref.get()

known_face_encodings = []
known_face_names = []
known_face_roles = []  # 👈 جديد

print("[INFO] تحميل صور المستخدمين من Firebase...")

for user_id, user_data in users.items():
    if all(k in user_data for k in ('firstName', 'fatherName', 'grandfatherName', 'familyName', 'image', 'role')):
        try:
            image_data = base64.b64decode(user_data['image'])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(image)

            encoding = face_recognition.face_encodings(image_np)
            if encoding:
                full_name = f"{user_data['firstName']} {user_data['fatherName']} {user_data['grandfatherName']} {user_data['familyName']}"
                known_face_encodings.append(encoding[0])
                known_face_names.append(full_name)
                known_face_roles.append(user_data['role'])  # 👈 حفظ الدور
                print(f"تم تحميل: {full_name} ({user_data['role']})")
        except Exception as e:
            print(f"[خطأ] في تحميل {user_id}: {e}")

print(f"[INFO] تم تحميل {len(known_face_names)} وجهًا معرفًا بنجاح")

video_capture = cv2.VideoCapture(0)
print("[INFO] بدأ التعرف على الوجوه...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "غير معروف"
        color = (0, 0, 255)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        if True in matches:
            idx = matches.index(True)
            name = known_face_names[idx]
            role = known_face_roles[idx]

            # 🟦 إذا الدور "patient" يكون المربع أزرق
            if role == "patient":
                color = (255, 0, 0)  # أزرق
            else:
                color = (0, 255, 0)  # أخضر

        # إعادة الإحداثيات لحجم الصورة الأصلي
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        frame = draw_arabic_text(frame, name, left, top - 30, color=color)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

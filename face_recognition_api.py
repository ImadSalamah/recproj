import base64
import io
import os
import json
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition

# ✅ تحميل مفتاح Firebase من متغير بيئة (SECRET)
service_account_info = json.loads(os.getenv("SERVICE_ACCOUNT_JSON"))
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://dcms-aaup-6e1e4-default-rtdb.firebaseio.com/'
})

# ✅ تحميل الوجوه من Firebase
ref = db.reference('users')
users = ref.get()
known_face_encodings = []
known_face_names = []

for user_id, user_data in users.items():
    if all(k in user_data for k in ('firstName', 'fatherName', 'grandfatherName', 'familyName', 'image')):
        try:
            if not user_data['image'].strip().startswith('/9j/'):
                print(f"Skipping user {user_id}: invalid base64 JPEG")
                continue

            image_data = base64.b64decode(user_data['image'])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(image)

            face_locations = face_recognition.face_locations(image_np)
            if not face_locations:
                print(f"No face found for user {user_id}")
                continue

            encoding = face_recognition.face_encodings(image_np, face_locations, num_jitters=1)
            if encoding:
                full_name = f"{user_data['firstName']} {user_data['fatherName']} {user_data['grandfatherName']} {user_data['familyName']}"
                known_face_encodings.append(encoding[0])
                known_face_names.append(full_name)
        except Exception as e:
            print(f"Error loading user {user_id}: {e}")

# ✅ إعداد Flask مع CORS
app = Flask(__name__)
CORS(app)

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        img_base64 = data['image']
        image_data = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        face_locations = face_recognition.face_locations(image_np)
        if not face_locations:
            return jsonify({"faces": []})

        face_encodings = face_recognition.face_encodings(image_np, face_locations, num_jitters=1)

        results = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.5)
            name = "غير معروف"
            if True in matches:
                idx = matches.index(True)
                name = known_face_names[idx]

            results.append({
                "name": name,
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left
            })

        return jsonify({"faces": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✅ لتشغيل محليًا أو على سيرفر
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

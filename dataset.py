import cv2
import os
import imutils
from datetime import datetime
import time

# Fungsi untuk menyimpan wajah ke folder dataset
def save_face(frame, box, name, save_dir="dataset_webcam"):
    if not os.path.exists(save_dir):  # Buat folder dataset jika belum ada
        os.makedirs(save_dir)
    person_dir = os.path.join(save_dir, name)  # Folder khusus untuk setiap orang
    if not os.path.exists(person_dir):  # Buat folder jika belum ada
        os.makedirs(person_dir)
    
    # Potong wajah dari frame
    top, right, bottom, left = box
    face_image = frame[top:bottom, left:right]
    
    # Buat nama file unik menggunakan timestamp
    filename = os.path.join(person_dir, f"{name}{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
    cv2.imwrite(filename, face_image)
    print(f"[INFO] Wajah disimpan di: {filename}")

# Inisialisasi webcam
print("[INFO] Mulai menangkap gambar wajah...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)  # Waktu pemanasan webcam

name = input("Masukkan nama untuk wajah ini: ")  # Nama orang untuk folder dataset

# Loop untuk menangkap gambar
while True:
    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Tidak dapat membaca dari webcam")
        break
    
    frame = imutils.resize(frame, width=500)  # Resize untuk efisiensi
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi wajah
    boxes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = boxes.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Gambar kotak di wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        save_face(frame, (y, x+w, y+h, x), name)  # Simpan wajah
    
    # Tampilkan video
    cv2.imshow("Capture Faces - Tekan 'q' untiuk keluar", frame)
    
    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup proses
vs.release()
cv2.destroyAllWindows()
print("[INFO] Selesai menangkap gambar wajah.")
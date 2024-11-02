import os
import cv2

# 이미지 저장 폴더 경로
folder_path = "C:\\Users\\User\\KT_Project\\M4\\img"
# 이미지 저장 폴더가 없다면 폴더 생성
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print('my_face 폴더를 생성합니다.')

# 웹캠으로 얼굴 사진을 찍어 저장하는 함수
def capture_owner_images(num_images=180):  # num_images에 숫자를 입력한 만큼 이미지 저장
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0  # count를 0으로 초기화
    while count < num_images:
        _, frame = cap.read()

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        face_found = False  # 얼굴 탐지 여부를 기록하는 플래그

        # 탐지된 얼굴에서 좌표를 가져오는 반복문
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (512, 512))
            face_file = os.path.join(folder_path, f"taeeon_face2_{count}.jpg")
            cv2.imwrite(face_file, resized_face)
            count += 1
            face_found = True  # 얼굴이 탐지됨을 기록
            cv2.imshow('Captured Face', resized_face)

        # 얼굴이 탐지되지 않았더라도 이미지를 저장
        if not face_found and count < num_images:
            cv2.imwrite(os.path.join(folder_path, f"image_{count}.jpg"), frame)
            count += 1

        # 작은 지연 추가 (예: 50ms)
        cv2.waitKey(50)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_owner_images()
print('저장 완료')

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

sticker_image = cv2.imread('01.png', cv2.IMREAD_UNCHANGED)

image_count = 0
captured_images = []

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(300, 300))

        for (x, y, w, h) in faces:
            alpha_channel = sticker_image[:, :, 3] / 255.0
            resized_overlay = cv2.resize(sticker_image[:, :, :3], (w, h))
            roi = frame[y:y+h, x:x+w]

            # Resize the sticker and overlay it on the face
            sticker_height = int(h * 0.5)
            sticker_width = int((sticker_height / resized_overlay.shape[0]) * resized_overlay.shape[1])
            sticker = cv2.resize(resized_overlay, (sticker_width, sticker_height))

            frame[y:y+sticker_height, x:x+sticker_width] = sticker
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        frame = cv2.flip(frame, 1)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            image_count += 1
            image_filename = f'captured_image_{image_count}.png'
            cv2.imwrite(image_filename, frame)
            print(f'Image captured and saved as {image_filename}')

            captured_image = cv2.imread(image_filename)
            captured_images.append(captured_image)

            # If 3 images are captured, exit the loop
            if image_count == 3:
                break

    else:
        break

cap.release()

# Merge captured images vertically and save the result
if captured_images:
    merged_image = np.vstack(captured_images)  # 수직으로 이미지 합치기
    cv2.imwrite('merged_image.png', merged_image)
    print("Vertical Merged image saved as merged_image_vertical.png")

else:
    print("No images captured.")

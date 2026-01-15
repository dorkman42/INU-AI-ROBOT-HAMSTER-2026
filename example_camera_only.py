"""
카메라 테스트 (로봇 없이)
- 모델 인식만 테스트
"""

from detector import Detector
import cv2

detector = Detector('keras_model.h5', 'labels.txt')
detector.open_camera()

print("카메라 테스트 중... (ESC로 종료)")

while True:
    frame = detector.read_frame()
    if frame is None:
        continue

    # 예측
    result = detector.predict(frame)

    if result:
        label = result['label']
        conf = result['confidence']

        # 화면에 표시
        color = (0, 255, 0) if label == 'safe' else (0, 0, 255)
        cv2.putText(frame, f"{label}: {conf:.0%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 콘솔 출력
        bar = "#" * int(conf * 20)
        print(f"\r{label}: {conf:.0%} {bar:<20}", end="")

    cv2.imshow("Detector", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

detector.close_camera()
cv2.destroyAllWindows()
print("\n종료")

"""
기본 예제: 햄스터 로봇 + AI 인식
- 기존 roboid 사용법 그대로 유지
- detector만 추가해서 인식
"""

from roboid import HamsterS
from detector import Detector
import time

# 로봇 연결
robot = HamsterS()

# AI 감지기 (카메라 + 모델)
detector = Detector('keras_model.h5', 'labels.txt')
detector.open_camera()

print("실행 중... (Ctrl+C로 종료)")

try:
    while True:
        # AI 인식
        result = detector.predict()

        if result:
            label = result['label']
            conf = result['confidence']
            print(f"\r{label}: {conf:.0%}", end="")

            # 인식 결과에 따라 로봇 제어
            if label == 'unsafe' and conf > 0.7:
                # 위험 감지: 정지 → 후진 → 회전
                robot.wheels(0, 0)
                robot.leds('red')
                robot.beep()
                robot.move_backward(3)
                robot.turn_left(90)
                robot.leds('green')

            elif label == 'safe' and conf > 0.7:
                # 안전: 전진
                robot.wheels(30, 30)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n종료")

finally:
    detector.close_camera()
    robot.wheels(0, 0)
    robot.leds('off')
    robot.dispose()

# 햄스터-S AI 세미나

Teachable Machine + 햄스터-S 로봇 AI 제어

## 설치

```bash
# uv 사용 (권장)
uv sync
```

## 빠른 시작

### 1. Teachable Machine에서 모델 학습
1. [Teachable Machine](https://teachablemachine.withgoogle.com/) 접속
2. Image Project → Standard image model
3. 클래스 추가 (예: `safe`, `unsafe`)
4. 학습 후 **Tensorflow → Keras** 모델 다운로드
5. `keras_model.h5`, `labels.txt` 프로젝트 폴더에 저장

### 2. 카메라 테스트 (로봇 없이)
```python
from detector import Detector

detector = Detector('keras_model.h5', 'labels.txt')
detector.open_camera()

while True:
    result = detector.predict()
    if result:
        print(f"{result['label']}: {result['confidence']:.0%}")
```

### 3. 로봇 + AI 인식
```python
from roboid import HamsterS
from detector import Detector

robot = HamsterS()
detector = Detector('keras_model.h5', 'labels.txt')
detector.open_camera()

while True:
    result = detector.predict()

    if result and result['confidence'] > 0.7:
        if result['label'] == 'unsafe':
            robot.wheels(0, 0)
            robot.beep()
            robot.move_backward(3)
            robot.turn_left(90)
        else:
            robot.wheels(30, 30)
```

## API

### Detector 클래스

```python
from detector import Detector

detector = Detector('keras_model.h5', 'labels.txt')
```

| 메서드 | 설명 |
|--------|------|
| `open_camera(index=0)` | 카메라 열기 |
| `close_camera()` | 카메라 닫기 |
| `read_frame()` | 프레임 읽기 |
| `predict(frame=None)` | 예측 (프레임 없으면 카메라에서 읽음) |
| `get_label(threshold=0.7)` | 라벨만 반환 |

### predict() 반환값

```python
{
    'label': 'unsafe',       # 라벨 이름
    'confidence': 0.95,      # 신뢰도 (0~1)
    'index': 0,              # 클래스 인덱스
    'all': {'safe': 0.05, 'unsafe': 0.95}  # 전체 확률
}
```

### 햄스터 로봇 API (roboid)

```python
from roboid import HamsterS

robot = HamsterS()
robot.wheels(left, right)    # 바퀴 속도 (-100 ~ 100)
robot.move_forward(cm)       # 전진
robot.move_backward(cm)      # 후진
robot.turn_left(degree)      # 좌회전
robot.turn_right(degree)     # 우회전
robot.leds(color)            # LED ('red', 'green', 'blue', 'off')
robot.beep()                 # 비프음
robot.dispose()              # 연결 해제
```

## 예제 파일

| 파일 | 설명 |
|------|------|
| `example_camera_only.py` | 카메라 인식 테스트 (로봇 없이) |
| `example_basic.py` | 로봇 + AI 기본 예제 |

## 참고

- [Teachable Machine](https://teachablemachine.withgoogle.com/)
- [roboid API 문서](https://github.com/RobomationLAB/Hamster-S_API_KR)

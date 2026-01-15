"""
Teachable Machine 모델 감지기
- 카메라에서 이미지를 읽어 모델로 예측
- 로봇 제어는 포함하지 않음
"""

import cv2
import numpy as np

__all__ = ['Detector']


class Detector:
    """Teachable Machine 모델 감지기"""

    def __init__(self, model_path='keras_model.h5', labels_path='labels.txt'):
        """
        Args:
            model_path: Keras 모델 파일 경로 (.h5)
            labels_path: 라벨 파일 경로 (.txt)
        """
        self.model = None
        self.labels = []
        self.camera = None
        self._load_model(model_path)
        self._load_labels(labels_path)

    def _load_model(self, path):
        """모델 로드"""
        try:
            import tf_keras as keras
            self.model = keras.models.load_model(path, compile=False)
            print(f"모델 로드 완료: {path}")
        except:
            try:
                import tensorflow.keras as keras
                self.model = keras.models.load_model(path, compile=False)
                print(f"모델 로드 완료: {path}")
            except Exception as e:
                print(f"모델 로드 실패: {e}")

    def _load_labels(self, path):
        """라벨 로드"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]
            print(f"라벨 로드 완료: {self.labels}")
        except Exception as e:
            print(f"라벨 로드 실패: {e}")
            self.labels = []

    def open_camera(self, index=0):
        """카메라 열기"""
        self.camera = cv2.VideoCapture(index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.camera.isOpened()

    def close_camera(self):
        """카메라 닫기"""
        if self.camera:
            self.camera.release()
            self.camera = None

    def read_frame(self):
        """카메라에서 프레임 읽기"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            return frame if ret else None
        return None

    def predict(self, frame=None):
        """
        예측 수행

        Args:
            frame: 이미지 (없으면 카메라에서 읽음)

        Returns:
            dict: {'label': str, 'confidence': float, 'index': int, 'all': dict}
            None: 예측 실패 시
        """
        if self.model is None:
            return None

        if frame is None:
            frame = self.read_frame()
        if frame is None:
            return None

        # 전처리: 224x224, -1~1 정규화
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) / 127.0) - 1
        img = img.reshape((1, 224, 224, 3))

        # 예측
        probs = self.model.predict(img, verbose=0)[0]
        top_idx = int(np.argmax(probs))

        return {
            'index': top_idx,
            'label': self.labels[top_idx] if top_idx < len(self.labels) else f"class_{top_idx}",
            'confidence': float(probs[top_idx]),
            'all': {self.labels[i] if i < len(self.labels) else f"class_{i}": float(p)
                    for i, p in enumerate(probs)}
        }

    def get_label(self, frame=None, threshold=0.7):
        """
        간단히 라벨만 반환 (threshold 이상일 때만)

        Args:
            frame: 이미지 (없으면 카메라에서 읽음)
            threshold: 신뢰도 임계값

        Returns:
            str: 라벨 (threshold 미만이면 None)
        """
        result = self.predict(frame)
        if result and result['confidence'] >= threshold:
            return result['label']
        return None

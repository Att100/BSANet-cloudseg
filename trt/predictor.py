import numpy as np
from PIL import Image

from .engine import TrtEngine

class TrtPredictor(object):
    def __init__(self, engine_path):
        self.engine = TrtEngine(engine_path)

    def run(self, img):
        img = self.preprocess(img)
        data = self.engine.infer(img)[0]
        pred = self.postprocess(data)
        return pred

    def preprocess(self, img: Image.Image):
        img = np.array(img.resize((320, 320))).transpose((2, 0, 1)) / 255
        img = (img.reshape((1, 3, 320, 320)) - 0.5) / 0.5
        return img.astype('float32')

    def postprocess(self, data: np.ndarray):
        pred = data.reshape((320, 320))
        pred = 1 / (1 + np.exp(-pred))  
        pred = Image.fromarray(np.uint8(pred * 255)).resize((300, 300))
        pred = (((np.array(pred) / 255) > 0.5) * 255).astype('int')
        return pred
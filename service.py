import numpy as np
from PIL import Image
import io

import bentoml
from bentoml.io import NumpyNdarray, Text

from transform import val_transform

runner = bentoml.onnx.get('mnist:latest').to_runner()

svc = bentoml.Service('mnist_classifier', runners = [runner]) # 1개 이상의 runner를 적용해줄 수 있다

def read_image(imgfile) :
    with open(imgfile, 'rb') as f :
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes))
    return val_transform()(image)

print(read_image('sample.jpg').shape)

@svc.api(input = Text(), output = NumpyNdarray())
def classify(input_series : str) -> np.ndarray :
    print(input_series)
    result = runner.run.run(read_image(input_series)).argmax(1)
    print(result)
    return result     
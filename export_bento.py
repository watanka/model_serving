import torch
import bentoml
from utils import read_yaml
from model import LightningModel
import onnxruntime


# config = read_yaml('config.yaml')
# plmodel = LightningModel(config)
# # TypeError: 'model' must be an instance of 'pl.LightningModule', got <class 'model.LightningModel'> instead.
# # error occurs when model has parameter to fill in.

# # convert to onnx
input_sample = torch.randn((1,28,28))
# plmodel.to_onnx('model.onnx', input_sample, export_params = True)

# infer with onnx
ort_session = onnxruntime.InferenceSession("model.onnx")
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ONNX Runtime will return a list of outputs
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0])



# save model and retrieve corresponding tag

bentoml.onnx.save_model
# tag = bentoml.pytorch_lightning.save_model('MNIST', LightningModel())

# # retrieve metadata with 'bentoml.models.get':
# metadata = bentoml.models.get(tag)

# # load the model 
# # model = bentoml.pytorch_lightning.load_model('MNIST:latest')

# # Run a given model under 'Runner' abstraction with 'to_runner'
# runner = bentoml.pytorch_lightning.get(tag).to_runner()
# runner.init_local()

# samp = torch.load('sample_gt_5.pt') # answer = 5

# runner.run(samp)
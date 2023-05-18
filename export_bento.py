import torch
import onnx
import onnxruntime
import bentoml
from utils import read_yaml
from model import LightningModel



config = read_yaml('config.yaml')
plmodel = LightningModel(config)
plmodel.load_from_checkpoint('mnist_weight.ckpt')
# # TypeError: 'model' must be an instance of 'pl.LightningModule', got <class 'model.LightningModel'> instead.
# # error occurs when model has parameter to fill in.

# # convert to onnx
input_sample = torch.randn((1,28,28))
plmodel.to_onnx('model.onnx', input_sample, export_params = True)

# # infer with onnx

# ort_session = onnxruntime.InferenceSession("model.onnx")
# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: input_sample.numpy()}
# # ONNX Runtime will return a list of outputs
# ort_outs = ort_session.run(None, ort_inputs)
# print(ort_outs[0])



# # save model and retrieve corresponding tag
onnx_model = onnx.load('model.onnx')

signatures = {
    'run' : {'batchable' : True,
             'batch_dim' : 0    
            }
}

bentoml.onnx.save_model('MNIST', onnx_model, signatures = signatures)
# tag = bentoml.pytorch_lightning.save_model('MNIST', LightningModel())

# # retrieve metadata with 'bentoml.models.get':
# metadata = bentoml.models.get(tag)

# load the model 
# model = bentoml.onnx.load_model('mnist:latest')

# # Run a given model under 'Runner' abstraction with 'to_runner'
runner = bentoml.onnx.get('mnist:latest').to_runner()
runner.init_local()

samp = torch.load('sample_gt_5.pt') # shape : (28, 28), answer = 5

print(dir(runner))

result = runner.run.run(samp.unsqueeze(0))
print(result.argmax(1))
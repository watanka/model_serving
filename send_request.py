import requests
import torch

samp = torch.load('sample_gt_5.pt')

requests.post(
    "http://0.0.0.0:3000/classify",
    headers = {'content-type' : 'application/json'},
    data = 'sample.jpg'
).text
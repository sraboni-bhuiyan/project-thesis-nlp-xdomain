""" import torch
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.version)
print(torch.__file__)
 """

import torch
print("torch:", torch.__version__)
print("cuda runtime in wheel:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    print("sm capability:", torch.cuda.get_device_capability(0))  
    # should be (8, 9) for Ada

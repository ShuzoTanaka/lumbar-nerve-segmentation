import torch
print(torch.cuda.is_available())  # True になるはず
print(torch.cuda.device_count())  # GPUの数
print(torch.cuda.get_device_name(0))  # GPU名

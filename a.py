import torch
print(torch.cuda.is_available())  # Trueであるべき
print(torch.cuda.device_count())  # 利用可能なGPUの数
print(torch.cuda.get_device_name(0))  # 利用可能なGPU名

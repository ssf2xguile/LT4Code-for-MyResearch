import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.tensor([0.1, 0.2]).cuda())

t_cpu = torch.tensor([0.1, 0.2])
t_gpu = t_cpu.to('cuda')
print(t_gpu)
print(torch.cuda.device_count())
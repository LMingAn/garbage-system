import torch
print(torch.cuda.is_available())
print(torch.__version__)        # 应该显示 2.5.1
print(torch.cuda.is_available()) # 应该显示 True
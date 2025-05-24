import soundata
import torch 
dataset = soundata.initialize("urbansound8k",data_home='/scratch/apadi089')
#dataset.download()
dataset.validate()
print(torch.cuda.get_device_name(0))


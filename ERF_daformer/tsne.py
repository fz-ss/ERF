# import numpy as np
# a=[[0.1,0.2,0.3],[0.7,0.3,0.4],[0.3,0.5,0.6]]
# a=np.array(a)
# split=list(a.argmax(axis=1))
# print(split)
# # total=[]
# # for i in range(len(split)):
# #     total.append(split.count(i))
# # print(total)
# np.random.shuffle(split)
# print(split)
# sample=[]
# for i in range(len(a[1])):
#     total=0
#     for index,value in enumerate(split):
#         if value==i and total<500:
#             sample.append(index) 
#             total+=1
#     # sample.append([index for index,value in enumerate(split) if value==j][:500])
# print(sample)

import torch

a=torch.tensor([1,2,3,4,5,6])
print(a)
with open("D:\\Codes\\ERF_daformer\\tsne.txt",'a') as of:
    of.write(str(a)+"\n")
idx = torch.randperm(a.size(0))
print(idx)
with open("D:\\Codes\\ERF_daformer\\tsne.txt",'a') as of:
    of.write(str(idx)+"\n")

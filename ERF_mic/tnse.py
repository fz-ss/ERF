# import numpy as np
# import torch
# ss_feature=torch.ones((2,28,19))
# ss_feature=list(ss_feature)
# ss_label=8*torch.ones((28))
# ss_label=list(ss_label)
# ss_feature = torch.cat(ss_feature, dim=0)
# ss_label = torch.cat(ss_label, dim=0)

# print("init_features size: " ,ss_feature.shape)
# print("init_labels size:",ss_label.shape)
# valid_indices = ss_label != 250
# filter_features = ss_feature[valid_indices]
# filter_labels = ss_label[valid_indices]
# print("fit_features size: ", filter_features.shape)
# print("fit_labels size:", filter_labels.shape)
# print("原本有多少类别",np.unique(ss_label))
# print("现在有多个类别",np.unique(filter_labels))
# # 测试类别数量：
# test = filter_labels.numpy()
# unique, counts = np.unique(test, return_counts=True)
# label_counts = dict(zip(unique, counts))
# print("每个类的数量分布：")
# print(label_counts)
# # 进行次阿一嗯
# n_samples = 10000
# print("sample dots:", n_samples)
# sample_indices = torch.randperm(filter_features.size(0))[:n_samples]
# ss_feature = filter_features[sample_indices]
# ss_label = filter_labels[sample_indices]
# print("last feaures:", ss_feature.shape)
# print("last featurrs: ", ss_label.shape)
# category_label ={
#     0: 'road',
#     1: 'sidewalk',
#     2: 'building',
#     3: 'wall',
#     4: 'fence',
#     5: 'pole',
#     6: 'light',
#     7: 'sign',
#     8: 'vegetation',
#     9: 'terrain',
#     10: 'sky',
#     11: 'person',
#     12: 'rider',
#     13: 'car',
#     14: 'truck',
#     15: 'bus',
#     16: 'train',
#     17: 'motorcycle',
#     18: 'bicycle',
# }
# # from tsnecuda import TSNE
# from sklearn.manifold import TSNE
# # X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)

# import matplotlib.pyplot as plt

# # 条参
# embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(ss_feature)
# print("ender tsne")
# # visual
# plt.figure(figsize=(12, 10))
# # 初版可用
# # scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=ss_label,
# #                       cmap='tab20', alpha=0.6, edgecolors='w', linewidths=0.5)
# # 自己调整
# scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=ss_label,
#                       cmap='tab20',
#                       alpha=0.6,
#                       # marker='o',
#                       s= 10,
#                       # edgecolors='w',
#                       # linewidths=0.5
#                       )
# print("总共有多华沙个label", len(np.unique(ss_label)))
# # exit()
# # 隐藏坐标
# # plt.axis("off")
# # 改变刻度标签文字
# colorbar = plt.colorbar(scatter, orientation='horizontal', pad=0.15, aspect=40)
# # colorbar.ax.tick_params(labelsize=8)
# # ticks = colorbar.get_ticks()
# colorbar.set_ticks(np.arange(min(ss_label), max(ss_label)+1))
# colorbar.set_ticklabels([category_label.get(label, "") for label in range(min(ss_label),
#                                                                          max(ss_label)+1)])
# #  txt坐标
# # yy_label = [category_label[t] for t in ticks]
# # print(yy_label)
# # exit()
# #  txt文字
# # 设置文字
# # for label in colorbar.ax.get_yticklabels():
# #     label.set_verticalalignment('center')
# #     label.set_horizontalalignment('right')
# plt.title("t-sne visual")
# plt.savefig('/media/ailab/data/syn/MIC/pred_ss/tnse/MIC.png', dpi=300)
# plt.show()
a = [0,1,2,3,4]
print(a[1,2,3])
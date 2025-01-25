# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Modification of config and checkpoint to support legacy models
# - Add inference mode and HRDA output flag
import os 
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import sys
sys.path.append('/media/ailab/data/syn/MIC')
import argparse


import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    cfg.data.val.pipeline[1]['img_scale'] = tuple(
        cfg.data.val.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', default='/media/ailab/data/syn/MIC/configs/mic/synHR2csHR_mic_hrda.py',help='test config file path')
    parser.add_argument('--checkpoint', default='/media/ailab/data/syn/MIC/work_dirs/MIC_pth/MIC_SYN.pth',help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=[
            'same',
            'whole',
            'slide',
        ],
        default='same',
        help='Inference mode.')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        # default=True,
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        default=True,
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', 
        # default='/media/ailab/data/syn/MIC/pred_ss/MIC_gta',
        help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        # default='/media/ailab/data/syn/MIC/pred_ss/MIC_gta',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
        #                           efficient_test, args.opacity)
        ss_image = torch.randperm(500)[:40] ###40
        # ss_image=[100,  27, 229, 478,  89,  33, 392, 447, 144,  23, 247, 426, 292, 412,
        #  58, 280, 269, 200, 360, 367, 135, 371, 164,  52, 170,  38, 175, 159,
        # 253, 475, 469, 231, 448, 473, 347,  90,  72,  75,  78, 383]

        print(ss_image)
        with open("/media/ailab/data/syn/MIC/pred_ss/tnse/choose.txt",'a') as of:
            of.write(str(ss_image)+'\n')



        ss_feature = single_gpu_test(model, data_loader,ss_image, args.show, args.show_dir,
                                  efficient_test, args.opacity)
        
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            # res = dataset.evaluate(outputs, args.eval, **kwargs)
            # print([k for k, v in res.items() if 'IoU' in k])
            # print([round(v * 100, 1) for k, v in res.items() if 'IoU' in k])
            ss_label=dataset.evaluate(ss_feature, ss_image, args.eval, **kwargs) #2097152
            import numpy as np
            
            ss_feature = torch.cat(ss_feature, dim=0)
            ss_label = torch.cat(list(torch.tensor(ss_label)), dim=0)
            
            print("init_features size: " ,ss_feature.shape)
            print("init_labels size:",ss_label.shape)
            valid_indices = ss_label != 250
            filter_features = ss_feature[valid_indices]
            filter_labels = ss_label[valid_indices]
            print("fit_features size: ", filter_features.shape)
            print("fit_labels size:", filter_labels.shape)
            print("原本有多少类别",np.unique(ss_label))
            print("现在有多个类别",np.unique(filter_labels))
            # 测试类别数量：
            test = filter_labels.numpy()
            unique, counts = np.unique(test, return_counts=True)
            label_counts = dict(zip(unique, counts))
            print("每个类的数量分布：")
            print(label_counts)
            # 进行次阿一嗯
            n_samples = 1000
            print("sample dots:", n_samples*(len(np.unique(filter_labels))-1))
            # sample_indices = torch.randperm(filter_features.size(0))[:10000]
            inx = torch.randperm(ss_label.size(0))
            sample_indices=[]
            for i in range(19):
                total=0
                for index in inx:
                    if ss_label[index]==i and total<n_samples:
                        sample_indices.append(index)
                        total+=1
                    elif total==1000:break
            print(len(sample_indices))
            with open("/media/ailab/data/syn/MIC/pred_ss/tnse/choose.txt",'a') as of:
                 of.write(str(sample_indices)+'\n')
            # sample_indices=


            ss_feature = filter_features[sample_indices]
            ss_label = filter_labels[sample_indices]
            print("last feaures:", ss_feature.shape)
            print("last featurrs: ", ss_label.shape)
            print("选择后有多少类别",np.unique(ss_label))
            test2 = ss_label.numpy()
            unique2, counts2 = np.unique(test2, return_counts=True)
            label_counts2 = dict(zip(unique2, counts2))
            print("选择后每个类的数量分布：")
            print(label_counts2)
            category_label ={
                0: 'Road',
                1: 'S.walk',
                2: 'Build',
                3: 'Wall',
                4: 'Fence',
                5: 'Pole',
                6: 'Light',
                7: 'Sign',
                8: 'Veg',
                9: 'Terrain',
                10: 'Sky',
                11: 'Person',
                12: 'Rider',
                13: 'Car',
                14: 'Truck',
                15: 'Bus',
                16: 'Train',
                17: 'M.bike',
                18: 'Bike',
            }
            # from tsnecuda import TSNE
            from sklearn.manifold import TSNE
            # X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
            
            import matplotlib.pyplot as plt
            ss_feature=ss_feature.cpu() 
            
            # 条参
            embeddings = TSNE(n_components=2, perplexity=25, learning_rate=100).fit_transform(ss_feature)
            print("ender tsne")
            # visual
            plt.figure(figsize=(12, 10))
            # 初版可用
            # scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=ss_label,
            #                       cmap='tab20', alpha=0.6, edgecolors='w', linewidths=0.5)
            # 自己调整
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=ss_label,
                                  cmap='tab20',
                                  alpha=0.6,
                                  # marker='o',
                                  s= 10,
                                  # edgecolors='w',
                                  # linewidths=0.5
                                  )
            print("总共有多少个label", len(np.unique(ss_label)))
            # exit()
            # 隐藏坐标
            # plt.axis("off")
            # 改变刻度标签文字
            colorbar = plt.colorbar(scatter, orientation='horizontal', pad=0.15, aspect=40)
            # colorbar.ax.tick_params(labelsize=8)
            # ticks = colorbar.get_ticks()
            colorbar.set_ticks(np.arange(min(ss_label), max(ss_label)+1))
            colorbar.set_ticklabels([category_label.get(label, "") for label in range(min(ss_label),
                                                                                     max(ss_label)+1)])
            #  txt坐标
            # yy_label = [category_label[t] for t in ticks]
            # print(yy_label)
            # exit()
            #  txt文字
            # 设置文字
            # for label in colorbar.ax.get_yticklabels():
            #     label.set_verticalalignment('center')
            #     label.set_horizontalalignment('right')
            plt.title("t-sne visual")
            plt.savefig('/media/ailab/data/syn/MIC/pred_ss/tnse/MIC5.png', dpi=300)
            plt.show()
            
            


if __name__ == '__main__':
    main()

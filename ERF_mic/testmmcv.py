import numpy as np
import torch

from mmcv.ops import box_iou_rotated


def check_installation():
    """Check whether mmcv-full has been installed successfully."""
    np_boxes1 = np.asarray(
        [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
         [7.0, 7.0, 8.0, 8.0, 0.4]],
        dtype=np.float32)
    np_boxes2 = np.asarray(
        [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
         [5.0, 5.0, 6.0, 7.0, 0.4]],
        dtype=np.float32)
    boxes1 = torch.from_numpy(np_boxes1)
    boxes2 = torch.from_numpy(np_boxes2)

    # test mmcv-full with CPU ops
    box_iou_rotated(boxes1, boxes2)

    # test mmcv-full with both CPU and CUDA ops
    if torch.cuda.is_available():
        boxes1 = boxes1.cuda()
        boxes2 = boxes2.cuda()

        box_iou_rotated(boxes1, boxes2)


if __name__ == '__main__':
    check_installation()
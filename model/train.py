import os
import numpy as np
import torch
import argparse
import random
from torch.utils.data import Dataset
# from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import sys

from PIL import Image
# import pandas as pd
from torchvision import transforms
from scipy.stats import multivariate_normal
import skimage.io
import skimage.transform
import skimage.color
import skimage
from D3Pose import D3Pose


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    # parser.add_argument(
    #     "-cd", "--contextDataset", type=str,
    #     default='D:/Tianma/dataset/Pre_process/train_tensor/',
    #     help="Training dataset"
    # )

    parser.add_argument(
        "-td", "--testing_Data", type=str, default='/home/hongji/Documents/data_copy/test/feature_maps',
        help="testing dataset"
    )

    parser.add_argument(
        "-d", "--Training_Data", type=str, default='/home/hongji/Documents/data_copy/train/feature_maps',
        help="Training dataset"
    )
    parser.add_argument("-e", "--epochs", default=1000000, type=int, help="Number of epochs (default: %(default)s)", )
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n", "--num-workers", type=int, default=8, help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size", type=int, nargs=2, default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=80, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=90, help="Test batch size (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save_path", type=str, default="/home/hongji/Documents/save", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument("--clip_max_norm", default=1.0, type=float,
                        help="gradient clipping max norm (default: %(default)s")

    parser.add_argument("--checkpoint",
                        default="",  # ./train0008/18.ckpt
                        type=str, help="Path to a checkpoint")

    args = parser.parse_args(argv)
    return args


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, min_side=256, max_side=256):
        # image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 256 - rows
        pad_h = 256 - cols

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        # annots *= scale

        return torch.from_numpy(new_image), scale


class myDataset(Dataset):

    def __init__(self, root, transform):
        # self.df = pd.read_csv(root)
        self.clipTensor = []

        for pt in os.listdir(root):
            pt_path = os.path.join(root, pt)
            self.clipTensor.append(pt_path)

        self.transform = transform

    def __getitem__(self, index):
        spatial_feature_map_path = self.clipTensor[index]

        split_string = spatial_feature_map_path.split('/')
        parts = spatial_feature_map_path.split('_')

        clip_index = parts[7]
        clip_index = clip_index.split('.')[0]

        pt_name = split_string[len(split_string) - 1]
        gt_name = pt_name.replace('image_feat', '').replace('.pt', '.npy')
        gt_name = f'clip{clip_index}{gt_name}'
        folder_path = '/'.join(split_string[:-2])

        gt_folder_path = os.path.join(folder_path, 'gt')
        gt_path = os.path.join(gt_folder_path, gt_name)

        spatial_feature_map = torch.load(spatial_feature_map_path, map_location=lambda storage, loc: storage)
        spatial_feature_map = spatial_feature_map.view(30, 200, 192)

        GT_npy = torch.from_numpy(np.array(np.load(gt_path), dtype='f'))

        return spatial_feature_map, GT_npy, GT_npy

    def __len__(self):
        return len(self.clipTensor)


def train_one_epoch(model, train_dataloader, optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device
    start = time.time()
    # accu_num = torch.zeros(1).to(device)
    sample_num = 0

    for i, d in enumerate(train_dataloader):

        Images, srcGT, GT_npy = d
        Images = Images.to(device)
        srcGT = srcGT.to(device)
        GT_npy = GT_npy.to(device)
        optimizer.zero_grad()
        sample_num += Images.shape[0]

        out_net = model(Images, srcGT)

        # compare 1 - 31 rows in GT with 0 - 30 in output
        out_net_clean = out_net[:, :30, :]
        GT_clean = GT_npy[:, 1:, :]

        loss_function = torch.nn.MSELoss(reduction='mean')
        out_criterion = loss_function(out_net_clean, GT_clean)
        out_criterion.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 200 == 0:
            enc_time = time.time() - start
            start = time.time()
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(Images)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion.item():.4f} |'
                # f'\tacc: {accu_num.item() / sample_num:.4f} |'
                f"\ttime: {enc_time:.1f}"
            )


def validate_epoch(epoch, test_dataloader, model):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    loss_function = torch.nn.MSELoss(reduction='mean')
    # accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0

    with torch.no_grad():
        for d in test_dataloader:
            Images, GT, GT_npy = d
            sample_num += Images.shape[0]
            out_net = model(Images.to(device), GT.to(device))
            out_net_clean = out_net[:, :30, :]
            GT_clean = GT_npy[:, 1:, :]

            out_criterion = loss_function(out_net_clean, GT_clean.to(device))

            loss.update(out_criterion)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        # f'\tacc: {accu_num.item() / sample_num:.4f} |'
    )
    return loss.avg


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose([Resizer()])

    # test_transforms = transforms.Compose([Resizer()])
    print('loading datasets')
    train_dataset = myDataset(args.Training_Data, train_transforms)
    test_dataset = myDataset(args.testing_Data, train_transforms)
    print('finish loading datasets')
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        # pin_memory=(device == "cuda"),
    )

    net = D3Pose()
    net = net.to(device)

    # print('GPU:',torch.cuda.device_count())
    #
    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    # criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        new_state_dict = checkpoint["state_dict"]
        # new_state_dict = OrderedDict()

        # for k, v in checkpoint["state_dict"].items():
        #     # if 'gaussian_conditional' in k:
        #     #     new_state_dict[k]=v
        #     #     print(k)
        #     #     continue
        #     # if 'module' not in k:
        #     k = k[7:]
        #     # else:
        #     #     k = k.replace('features.module.', 'module.features.')
        #     new_state_dict[k]=v

        net.load_state_dict(new_state_dict)

        optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(net, train_dataloader, optimizer,
                        epoch,
                        args.clip_max_norm,
                        )
        loss = validate_epoch(epoch, test_dataloader, net)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best and epoch % 1 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                args.save_path + str(epoch) + '.ckpt'
            )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])

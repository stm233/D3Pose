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
        "-td", "--testing_Data", type=str, default='/home/hongji/Documents/h36m_data_no_sw/validation/feature_maps',
        help="testing dataset"
    )

    parser.add_argument(
        "-d", "--Training_Data", type=str, default='/home/hongji/Documents/h36m_data_no_sw/train/feature_maps',
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
        "--batch-size", type=int, default=40, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=40, help="Test batch size (default: %(default)s)",
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
                        default="",  # ./train0008/10.ckpt
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
        pt_name = split_string[len(split_string) - 1]
        gt_name = pt_name.replace('.pt','.npy')

        folder_path = '/'.join(split_string[:-2])

        gt_folder_path = os.path.join(folder_path, 'gt')
        gt_path = os.path.join(gt_folder_path, gt_name)

        with torch.no_grad():
            spatial_feature_map = torch.load(spatial_feature_map_path, map_location=lambda storage, loc: storage)
            spatial_feature_map = spatial_feature_map.view(30, 200, 192)
            spatial_feature_map.requires_grad = False

            GT_npy = torch.from_numpy(np.array(np.load(gt_path), dtype='f'))
            GT_npy.requires_grad = False
            # print(GT_npy.dtype)

        # mu + sigma * random_number

        # GT_npy = random_tensor

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

        random_tensor = GT_npy + 0 * torch.randn_like(GT_npy) * optimizer.param_groups[0]['lr']

        # Set the first row to zero
        random_tensor[:, 0] = 0
        # random_tensor[:, 0, :] = 0

        srcGT = random_tensor

        Images = Images.to(device)
        srcGT = srcGT.to(device)
        GT_npy = GT_npy.to(device)
        optimizer.zero_grad()
        sample_num += Images.shape[0]

        out_net = model(Images, srcGT)

        # compare 1 - 31 rows in GT with 0 - 30 in output
        out_net_clean = out_net[:, :30, :]
        GT_clean = GT_npy[:, 1:, :]

        beta_out = out_net_clean[:, 29, -10:]
        pose_out = out_net_clean[:, :, :72]
        pose_first_frame_out = pose_out[:, 0, :]

        gt_beta = GT_clean[:, 29, -10:]
        gt_pose = GT_clean[:, :, :72]
        pose_first_frame_gt = gt_pose[:, 0, :]

        loss_function_beta = torch.nn.MSELoss(reduction='mean')
        loss_function_pose = torch.nn.MSELoss(reduction='mean')
        loss_function_pose_first = torch.nn.MSELoss(reduction='mean')
        loss_function_GTs = torch.nn.MSELoss(reduction='mean')

        out_criterion_beta = loss_function_beta(beta_out.to(device), gt_beta.to(device))
        out_criterion_pose = loss_function_pose(pose_out.to(device), gt_pose.to(device))
        out_criterion_pose_first = loss_function_pose_first(pose_first_frame_out.to(device), pose_first_frame_gt.to(device))
        out_criterion_GTs = loss_function_GTs(srcGT.to(device),GT_npy.to(device))

        alpha = 1  # weight for the first loss
        beta = 0  # weight for the second loss
        theta = 0

        combined_loss = alpha * out_criterion_pose + beta * out_criterion_beta + theta * out_criterion_pose_first
        # combined_loss = the * out_criterion_pose_first

        combined_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 300 == 0:
            enc_time = time.time() - start
            start = time.time()
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(Images)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {combined_loss.item():.7f} |'
                f'\tbeta_Loss: {out_criterion_beta.item():.7f} |'
                f'\tpose_Loss: {out_criterion_pose.item():.7f} |'
                f'\tGT_Loss: {out_criterion_GTs.item():.7f} |'
                f'\tpose_First_Frame_Loss: {out_criterion_pose_first.item():}'
                # f'\tacc: {accu_num.item() / sample_num:.4f} |'
                f"\ttime: {enc_time:.1f}"
            )
        if i > 300:
            break


def validate_epoch(epoch, test_dataloader, model):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    # loss_function = torch.nn.MSELoss(reduction='mean')
    # accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0

    with torch.no_grad():
        for d in test_dataloader:
            Images, GT, GT_npy = d
            sample_num += Images.shape[0]
            out_net = model(Images.to(device), GT_npy.to(device))

            out_net_clean = out_net[:, :30, :]
            GT_clean = GT_npy[:, 1:, :]

            beta_out = out_net_clean[:, 29, -10:]
            pose_out = out_net_clean[:, :, :72]
            pose_first_frame_out = pose_out[:, 0, :]

            gt_beta = GT_clean[:, 29, -10:]
            gt_pose = GT_clean[:, :, :72]
            pose_first_frame_gt = gt_pose[:, 0, :]

            loss_function_beta = torch.nn.MSELoss(reduction='mean')
            loss_function_pose = torch.nn.MSELoss(reduction='mean')
            loss_function_pose_first = torch.nn.MSELoss(reduction='mean')

            out_criterion_beta = loss_function_beta(beta_out.to(device), gt_beta.to(device))
            out_criterion_pose = loss_function_pose(pose_out.to(device), gt_pose.to(device))
            out_criterion_pose_first = loss_function_pose_first(pose_first_frame_out.to(device), pose_first_frame_gt.to(device))

            alpha = 1  # weight for the first loss
            beta = 0  # weight for the second loss
            theta = 0

            combined_loss = alpha * out_criterion_pose + beta * out_criterion_beta + theta * out_criterion_pose_first

            loss.update(combined_loss)
            # if sample_num > 1200:
            #     break

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.7f} |"
        f'\tbeta_Loss: {out_criterion_beta.item():.7f} |'
        f'\tpose_Loss: {out_criterion_pose.item():.7f} |'
        f'\tpose_First_Frame_Loss: {out_criterion_pose_first.item():}'
        # f'\tacc: {accu_num.item() / sample_num:.4f} |'
    )
    return loss.avg


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose([])

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

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=4)
    # criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        new_state_dict = checkpoint["state_dict"]

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

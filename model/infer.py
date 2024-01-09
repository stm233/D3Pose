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
from model.D3Pose import*
from PIL import Image
# import pandas as pd
from torchvision import transforms
from scipy.stats import multivariate_normal
import skimage.io
import skimage.transform
import skimage.color
import skimage


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe(predicteds, targets):
    assert predicteds.shape == targets.shape

    targets = targets.cpu().numpy()
    predicteds = predicteds.cpu().numpy()
    output = 0
    for i in range(targets.shape[0]):
        target = targets[i]
        predicted = predicteds[i]
        muX = np.mean(target, axis=1, keepdims=True)
        muY = np.mean(predicted, axis=1, keepdims=True)

        X0 = target - muX
        Y0 = predicted - muY

        normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
        normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

        X0 /= normX
        Y0 /= normY

        H = np.matmul(X0.transpose(0, 2, 1), Y0)
        U, s, Vt = np.linalg.svd(H)
        V = Vt.transpose(0, 2, 1)
        R = np.matmul(V, U.transpose(0, 2, 1))

        sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
        V[:, :, -1] *= sign_detR
        s[:, -1] *= sign_detR.flatten()
        R = np.matmul(V, U.transpose(0, 2, 1))

        tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

        a = tr * normX / normY
        t = muX - a * np.matmul(muY, R)

        predicted_aligned = a * np.matmul(predicted, R) + t
        result = np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1),
                         axis=len(target.shape) - 2)
        output = output + result
    output = output / targets.shape[0]
    return np.mean(output)


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

    parser.add_argument(
        "-td", "--testing_Data", type=str, default='/media/hongji/4T/Downloads/3DPW/preprocess_dataset/validation', help="testing dataset"
    )
    # /media/imaginarium/2T   '/media/imaginarium/12T_2/train/

    parser.add_argument(
        "-n", "--num-workers", type=int, default=8, help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Test batch size (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save_path", type=str, default="./save/", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument("--clip_max_norm", default=1.0, type=float,
                        help="gradient clipping max norm (default: %(default)s")

    parser.add_argument("--checkpoint",
                        default="./save/20.ckpt",  # ./train0008/18.ckpt
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

        for vid in os.listdir(root):
            vid_path = os.path.join(root, vid)
            for clip in os.listdir(vid_path):
                clip_path = os.path.join(vid_path, clip)
                for file in os.listdir(clip_path):
                    if file.endswith('.pt'):
                        pt_path = os.path.join(clip_path, file)
                        self.clipTensor.append(pt_path)
                        break

        self.transform = transform

    def __getitem__(self, index):
        spatial_feature_map_path = self.clipTensor[index]
        #GT_name = spatial_feature_map_path.split('/')[-2] + '.npy'
        #clip_path = spatial_feature_map_path[:-14]

        clip_dir = os.path.dirname(spatial_feature_map_path)
        GT_path = None

        for file in os.listdir(clip_dir):
            if file.endswith('.npy'):
                GT_path = os.path.join(clip_dir, file)
                break

        spatial_feature_map = torch.load(spatial_feature_map_path, map_location=lambda storage, loc: storage)
        spatial_feature_map = spatial_feature_map.view(30,200,192)

        GT_npy = torch.from_numpy(np.array(np.load(GT_path), dtype='f'))

        # GT_npy = GT_npy * 1 / (100 * 1)
        # heatmaps = GT_npy

        return spatial_feature_map, GT_npy

    def __len__(self):
        return len(self.clipTensor)


def test_epoch(epoch, test_dataloader, model):
    model.eval()
    device = next(model.parameters()).device

    MSE = AverageMeter()
    MPJPE = AverageMeter()
    P_MPJPE = AverageMeter()
    loss_function = torch.nn.MSELoss(reduction='mean')
    sample_num = 0

    with torch.no_grad():
        for d in test_dataloader:
            images, GT = d
            images = images.to(device)
            # sample_num += Images.shape[0]
            # out_net = model(Images.to(device))

            start_token = torch.zeros(GT.shape, dtype=torch.float).to(device)
            input_seq = start_token
            input_seq[:, 0] = GT[:, 0]

            for frame in range(30):
                out_net = model(images, input_seq) # GT.to(device)
                # out_net2 = model(images, GT.to(device))

                input_seq[:,frame + 1] = GT[:,frame + 1]

            out = out_net
            out_net_clean = out[1:, :]
            GT_clean = GT[1:, :]

            out_criterion = loss_function(out_net_clean, GT_clean.to(device))
            MSE.update(out_criterion)

            print(
                f"Test epoch {epoch}: Average losses:"
                f"\tMSE: {MSE.avg:.5f} |"
                f"\tMPJPE: {MPJPE.avg:.3f} |"
                f"\tP_MPJPE: {P_MPJPE.avg:.3f} |"
                )

    return MSE.avg


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose([Resizer()])
    test_dataset = myDataset(args.testing_Data, train_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        # pin_memory=(device == "cuda"),
    )

    net = D3Pose()
    net = net.to(device)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        new_state_dict = checkpoint["state_dict"]

        net.load_state_dict(new_state_dict)

    loss = test_epoch(0, test_dataloader, net)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])
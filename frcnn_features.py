import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as M
from Progress import Progress
from utils import load_image_for_yolo
from sys import argv

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=800):
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_names = [x for x in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, x))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, i):
        image_path = os.path.join(self.image_dir, self.image_names[i])
        image_id = os.path.splitext(self.image_names[i])[0]

        image = load_image_for_yolo(image_path, self.image_size)
        image = torch.from_numpy(image).float()
        image /= 255.0

        return image_id, image

class ActivationHook:
    def __init__(self):
        self.activation = None

    def __call__(self, model, x, y):
        self.activation = torch.mean(y, dim=0)

def main():

    if len(argv) <= 1:
        print(f'Usage: {argv[0]} DATASET_PATH')
        print('')
        print('DATASET_PATH is assumed to lead to a directory with the following directories of images:')
        print('DATASET_PATH/images/train')
        print('DATASET_PATH/images/val')
        print('DATASET_PATH/images/test')
        print('')
        print('This script will output 3 new directories containint features extracted from Faster-RCNN:')
        print('DATASET_PATH/train_frcnn_features')
        print('DATASET_PATH/val_frcnn_features')
        print('DATASET_PATH/test_frcnn_features')
        return

    dataset_path = argv[1]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = M.detection.fasterrcnn_resnet50_fpn_v2(weights = M.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).to(device)
    model.eval()

    hook = ActivationHook()
    model.roi_heads.box_head.register_forward_hook(hook)

    train_data = ImageDataset(os.path.join(dataset_path, 'images/train'))
    train_dl = DataLoader(train_data, batch_size=1, num_workers=4)

    val_data = ImageDataset(os.path.join(dataset_path, 'images/val'))
    val_dl = DataLoader(val_data, batch_size=1, num_workers=4)

    test_data = ImageDataset(os.path.join(dataset_path, 'images/test'))
    test_dl = DataLoader(test_data, batch_size=1, num_workers=4)

    train_feat_dir = os.path.join(dataset_path, 'train_frcnn_features')
    if not os.path.exists(train_feat_dir): os.mkdir(train_feat_dir)

    val_feat_dir = os.path.join(dataset_path, 'val_frcnn_features')
    if not os.path.exists(val_feat_dir): os.mkdir(val_feat_dir)

    test_feat_dir = os.path.join(dataset_path, 'test_frcnn_features')
    if not os.path.exists(test_feat_dir): os.mkdir(test_feat_dir)

    with torch.no_grad():
        for image_id, images in Progress(train_dl):
            model(images.to(device))
            features = hook.activation.cpu().numpy()
            n = image_id[0] + '.np'
            out_path = os.path.join(train_feat_dir, n)
            features.tofile(out_path)

        for image_id, images in Progress(val_dl):
            model(images.to(device))
            features = hook.activation.cpu().numpy()
            n = image_id[0] + '.np'
            out_path = os.path.join(val_feat_dir, n)
            features.tofile(out_path)

        for image_id, images in Progress(test_dl):
            model(images.to(device))
            features = hook.activation.cpu().numpy()
            n = image_id[0] + '.np'
            out_path = os.path.join(test_feat_dir, n)
            features.tofile(out_path)

if __name__ == '__main__': main()

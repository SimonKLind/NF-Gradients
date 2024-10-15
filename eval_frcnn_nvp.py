import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as M
from utils import load_image_for_yolo
from realnvp import RealNVP
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

        image = load_image_for_yolo(image_path, self.image_size)
        image = torch.from_numpy(image).float()
        image /= 255.0

        return image

class ActivationHook:
    def __init__(self):
        self.activation = None

    def __call__(self, model, x, y):
        self.activation = torch.mean(y, dim=0)

def main():

    if len(argv) <= 1:
        print(f'Usage: {argv[0]} IMAGES_DIR')
        print('')
        print('This script will go through images inside IMAGES_DIR and display them')
        print('alongside their gradient image from a trained Normalizing flow.')
        print('The Normalizing Flow is assumed to have been produced by train_realnvp.py')
        return

    images_dir = argv[1]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = M.detection.fasterrcnn_resnet50_fpn_v2(weights = M.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).to(device)
    model.eval()

    hook = ActivationHook()
    model.roi_heads.box_head.register_forward_hook(hook)

    dataset = ImageDataset(images_dir)
    dl = DataLoader(dataset, batch_size=1, shuffle=True)

    nvp = RealNVP(1024, 512, 5).to(device)
    nvp.load_state_dict(torch.load('frcnn_nvp.pt'))
    nvp.eval()
    for p in nvp.parameters(): p.requires_grad = False

    # We apply gaussian noise to the input image, which further enhances the visual appeal
    # of the resulting gradient images.
    #
    # NOTE that this is purely for visualization purposes and was not used in the paper!
    best_noise = 0.004

    for images in dl:
        orig_images = images.to(device)
        gradimage = torch.zeros_like(images, device=device)
        for _ in range(10):
            images = orig_images + torch.randn_like(orig_images) * best_noise
            images.requires_grad = True
            model(images)

            log_Px, log_Pu, log_det = nvp(hook.activation.unsqueeze(0))
        
            loss = torch.mean(log_Px)
            if images.grad is not None: images.grad.zero_()
            loss.backward()

            gradimage += torch.abs(images.grad)

        gradimage /= 10.0
        gradimage = gradimage.squeeze().cpu().numpy()
        gradimage = gradimage.transpose(1, 2, 0)[:,:,::-1]
        gradimage = np.ascontiguousarray(gradimage)

        # Turn the gradient image into a floating-point grayscale image
        gray = np.mean(gradimage, axis=2)
        gray = np.log(gray + 1.0)
        gray -= np.min(gray)
        gray /= np.max(gray)
        gradimage[:,:,0] = gray
        gradimage[:,:,1] = gray
        gradimage[:,:,2] = gray

        image = images.squeeze().detach().cpu().numpy()
        image = image.transpose(1, 2, 0)[:,:,::-1]
        image = np.ascontiguousarray(image)

        im = np.hstack((image, gradimage))
        cv.imshow('test', im)
        key = cv.waitKey(0) & 0xff
        if key == ord('q'): break

if __name__ == '__main__': main()

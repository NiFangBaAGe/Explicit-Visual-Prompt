import argparse
import os
from PIL import Image

import torch
from torchvision import transforms
import models
import yaml
from mmcv.runner import get_dist_info, init_dist, load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(config['model']).cuda()
    if 'segformer' in config['model']['name']:
        checkpoint = load_checkpoint(model.encoder, args.model)
        model.encoder.PALETTE = checkpoint
        if args.prompt != 'none':
            print('loading prompt...')
            checkpoint = torch.load(args.prompt)
            model.encoder.backbone.prompt_generator.load_state_dict(checkpoint['prompt'])
            model.encoder.decode_head.load_state_dict(checkpoint['decode_head'])
    else:
        model.encoder.load_state_dict(torch.load(args.model), strict=False)

    # python demo.py --input defocus.png --model ./mit_b4.pth --prompt /home/user/project/prompting_weights/sota/imagenet/_train_segformer_evp_defocus_imagenet/prompt_epoch_last.pth --resolution 320,320 --gpu 0 --config configs/demo.yaml
    h, w = list(map(int, args.resolution.split(',')))
    img_transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
    ])

    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1, 1, 1])
    ])

    img = Image.open(args.input).convert('RGB')
    img = img_transform(img)
    img = img.cuda()

    pred = model.encoder.forward_dummy(img.unsqueeze(0))
    pred = torch.sigmoid(pred).view(1, w, h).cpu()

    transforms.ToPILImage()(pred).save(args.output)





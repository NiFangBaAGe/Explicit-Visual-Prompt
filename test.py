import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchsummary import summary
from mmcv.runner import get_dist_info, init_dist, load_checkpoint


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'sod':
        metric_fn = utils.calc_sod
        metric1, metric2, metric3, metric4 = 'f_max', 'mae', 's_max', 'e_max'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')


    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.encoder.forward_dummy(inp))

        # if eval_type is not None: # reshape for shaving-eval
        #     ih, iw = batch['inp'].shape[-2:]
        #
        #     shape = [batch['inp'].shape[0], ih, iw, 1]
        #     pred = pred.view(*shape) \
        #         .permute(0, 3, 1, 2).contiguous()
        #     batch['gt'] = batch['gt'].view(*shape) \
        #         .permute(0, 3, 1, 2).contiguous()

        # f1, auc, metric3 = metric_fn(pred, batch['gt'])
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    '''
    vit testing:
    python test.py --config configs/test/test-DHGAN_shadow.yaml  --model  ../prompting_weights/train/segformer/_train_segformer_sbu_imagenet/model_epoch_last.pth
    
    prompt-based testing:
    python test.py --config configs/test/test-DHGAN_shadow.yaml  --model save/_train_segformer_self_supervised/model_epoch_last.pth --prompt ../prompting_weights/train/self_supervised/_train_segformer_vfp_base_sbu_self_supervised/prompt_epoch_last.pth
    '''

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    # model = models.make(config['model']).cpu()
    # model.encoder.load_state_dict(torch.load(args.model, map_location='cpu'))
    model = models.make(config['model']).cuda()
    # model.encoder.load_state_dict(torch.load(args.model), strict=False)

    # if 'segformer' in args.model:
    if 'segformer' in config['model']['name']:
        print('loading public pretrain backbone...')
        checkpoint = load_checkpoint(model.encoder, args.model)
        model.encoder.PALETTE = checkpoint
        if args.prompt != 'none':
            print('loading prompt...')
            checkpoint = torch.load(args.prompt)
            model.encoder.backbone.prompt_generator.load_state_dict(checkpoint['prompt'])
            model.encoder.decode_head.load_state_dict(checkpoint['decode_head'])
    else:
        model.encoder.load_state_dict(torch.load(args.model), strict=False)

    # model.encoder.prompt_generator.pos_embed.load_state_dict(checkpoint_model['prompt_generator.pos_embed'], strict=False)


    metric1, metric2, metric3, metric4 = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))

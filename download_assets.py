import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import torch

from models import tensor_digest, verify_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--files', nargs='+', default=['dinov2_vits14_pretrain.pth'])
    args = parser.parse_args()
    entries = {x['file']: x for x in json.loads((Path(__file__).parent / 'assets/weights.json').read_text())}
    args.assets.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        if name not in entries:
            raise ValueError(f'Unknown asset: {name}')
        target = args.assets / name
        if target.exists():
            verify_weight(args.assets, name)
            print('Verified:', name)
            continue
        if name == 'saan_baid_pretrain.pth':
            raise RuntimeError('Download the official SAAN pretraining file from the link in assets/weights.json and place it in --assets')
        temporary = target.with_suffix(target.suffix + '.download')
        if temporary.exists():
            raise FileExistsError(temporary)
        if name == 'vgg19_relu4_1_imagenet.pth':
            urlretrieve('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', temporary)
            state = torch.load(temporary, map_location='cpu', weights_only=True)
            state = {key[len('features.'):]: value for key, value in state.items()
                     if key.startswith('features.') and int(key.split('.')[1]) < 21}
            if tensor_digest(state) != entries[name]['tensor_sha256']:
                raise ValueError('Unexpected ImageNet VGG19 feature weights')
            converted = target.with_suffix('.converted.pth')
            if converted.exists():
                raise FileExistsError(converted)
            torch.save(state, converted)
            converted.rename(target)
            temporary.unlink()
        else:
            urlretrieve(entries[name]['source'], temporary)
            temporary.rename(target)
        verify_weight(args.assets, name)
        print('Verified:', name)


if __name__ == '__main__':
    main()

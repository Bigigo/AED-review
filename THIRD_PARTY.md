# Implementation sources

## DINOv2

Frozen non-register DINOv2-S/14 from the [official Meta implementation](https://github.com/facebookresearch/dinov2/tree/85a2460), using `dinov2_vits14_pretrain.pth`. The code is obtained separately at the pinned revision; the Apache-2.0 licence is included in `licenses/`. The backbone itself is not redistributed here.

## MLSP

The fractional area pooling, 43 InceptionResNetV2 mixed features and wide pooling/scoring layout follow [MLSP](https://github.com/subpic/ava-mlsp/tree/dfb5c19b1e7139b1919ca1c1c766ca154aaad6be). The implementation is adapted to PyTorch and the AED objective. The MLSP code is MIT licensed; its notice is retained. The self-contained InceptionResNetV2 backbone subset derives from timm 0.9.2, `timm/models/inception_resnet_v2.py`, under Apache-2.0. Model tensor names and equations are retained while factory dependencies are removed. The MLVS-Net scoring front end and scalar head are specified separately in `dino_models.py`; they are not an unchanged MLSP model.

## NIMA

The AED adapter uses an ImageNet ResNet18, ten probability bins and the expected score in [0,1]. It trains on the scalar mid-rank target with Smooth L1, rather than an original rating-distribution objective. The ResNet backbone follows torchvision. The NIMA method is described in Talebi and Milanfar, *NIMA: Neural Image Assessment*.

## TANet

The adapter follows [TANet](https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment/tree/da7aebdb38ea8e2489ecd67e4f2f8a77cc3d23e0/code/TAD66K). It uses a frozen Places365 ResNet18 theme branch, a trainable MobileNetV2 aesthetic branch and an RGB relation branch. The AED implementation uses a sample-dependent 100x10 projection and the dropout settings encoded in the supplied source. Its objective, input size and training protocol follow AED. The upstream Apache-2.0 notice is included.

## SAAN

The adapter follows [BAID/SAAN](https://github.com/Dreemurr-T/BAID/tree/d5c9b8cddc1e1e419cdce71b50fc432a53f6fd57). It retains the General Aesthetic Branch, Style-Adaptive Aesthetic Branch, nonlocal block, 2x2 pooling and prediction head. General Aesthetic Branch parameters and VGG style parameters are frozen. BN running buffers in the General Aesthetic Branch still follow train/eval. Float32 AdaIN and attention accumulation, the external float32 sigmoid, and AED supervision are explicit numerical/training adaptations. No BAID dataset training is performed by this package. The authors' pretrained branch weights are obtained separately.

## HLA-GCN

The PyTorch adapter follows the equations in the [authors' Jittor implementation](https://github.com/shedy-pub/hlagcn-jittor/tree/5ee74a3f9222fa330979242516530521a766fefe). It uses an ImageNet ResNet50, three coordinate graph blocks, 25 latent nodes, and two ten-bin outputs. At 224x224 input, the coordinate grid is 7x7. The original reported grid/input and training recipe are not silently substituted for the supplied AED settings.

## Distribution

Upstream author names, source links, licence texts and copyright notices are attribution, not identifying information about this submission. They must be retained. The absence of a separate upstream licence in a source snapshot must not be interpreted as a blanket permission to redistribute that upstream project or its data. Pretrained weights and original upstream repositories are not embedded in this code package.

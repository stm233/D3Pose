# D3Pose - Double-Masked Encoder-Decoder model for video mesh recovery 

This research project aims to beat SOTA models on video mesh recovery with a novel cross-attention transformer-based model.

## Usage

install the following dependencies

```bash
python 3.8
pip install torch
pip install numpy
pip install torchvision
pip install scikit-image
pip install timm
pip install transformers
pip install pygame
pip install PyOpenGL PyOpenGL_accelerate
pip install pyrr
pip install chumpy
pip install opencv-python

```

## Pipeline

![Pipeline](assets/picture1.png)
Inspired by the [vanilla transformer](https://arxiv.org/pdf/1706.03762.pdf) in NLP. This paper utilizes an encoder-decoder architecture to strengthen temporal smoothness. The vanilla self-attention module is replaced by the swin transformer module.

![cross attention](assets/picture2.png)

## References

- check out [this](references/References.md) for a comprehensive list of SOTA model in mesh recovery.
- the code base is partially inspired by [this](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/transformer_from_scratch)

## To-dos

- train the basic cross-attention model on H36M with NeuralAnnot annotations
- train the basic cross-attention model on an expanded dataset
- designed the swin-transformer cross-attention model

## result log
- training on 3DPW
- during training, test epoch loss: 0.022, training epoch: 0.0036
- testing on test dataset with complete GT as input: 0.1377
- testing on test dataset with GT in loop as input: 0.14409
- testing on validation dataset with complete GT as input: 0.02028

#### Models

| Year | Model Name | Model-type | Temporal | Networks | output | interm | Multi-p* | Whole-body | Supervision |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 2023 | [Multi-view](https://arxiv.org/pdf/2212.05223.pdf) |  |  |  |  |  |  |  |  |
| 2023 | [DeFormer](https://openaccess.thecvf.com/content/CVPR2023/papers/Yoshiyasu_Deformable_Mesh_Transformer_for_3D_Human_Mesh_Recovery_CVPR_2023_paper.pdf) | Regression | - | transformer |  |  |  |  |  |
| 2023 | [POTTER](https://arxiv.org/pdf/2303.13357.pdf) | Regression | - | Vit |  |  |  |  |  |
| 2023 | [FeatER](https://arxiv.org/pdf/2205.15448.pdf) | Regression | - | Vit |  |  |  |  |  |
| 2023 | [Trace](https://arxiv.org/pdf/2306.02850.pdf) | Regression | - |  |  |  |  |  |  |
| 2023 | [HMR 2.0](https://arxiv.org/pdf/2305.20091.pdf) | Regression | - | ViT | params |  | x | x |  |
| 2023 | [OSX](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.pdf) | Regression | - | ViT | params |  | ✓ |  |  |
| 2023 | [SA-HMR](https://arxiv.org/pdf/2306.03847.pdf) | Regression | - | Trasformer | mesh |  | x | ✓ |  |
| 2023 | [CoorFormer](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Coordinate_Transformer_Achieving_Single-stage_Multi-person_Mesh_Recovery_from_Videos_ICCV_2023_paper.pdf) | Regression | STT | STT | params map |  | ✓ |  |  |
| 2022 | [FastMETRO](https://arxiv.org/pdf/2207.13820.pdf) | Regression | - | Transformer | mesh |  | x | .  ✓ |  |
| 2022 | [CLIFF](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Learning_Analytical_Posterior_Probability_for_Human_Mesh_Recovery_CVPR_2023_paper.html) | Regression | - | CNN | probability |  | x | ✓ |  |
| 2022 | [PyMAF-X](https://arxiv.org/pdf/2207.06400.pdf) | Regression | - | CNN | params |  | x | x |  |
| 2022 | [GLAMR](https://arxiv.org/pdf/2112.01524.pdf) | Regression |  |  |  |  |  |  |  |
| 2022 | [BEV](https://arxiv.org/pdf/2112.08274.pdf) | Regression | - | CNN | mesh map | - | ✓ | x |  |
| 2022 | [OCHMR](https://arxiv.org/pdf/2203.13349.pdf) | Regression | - | CNN | params |  | ✓ | x |  |
| 2022 | [MPS-Net](https://arxiv.org/abs/2203.08534) | Regression | HAFI | ResNet | params |  | x | x |  |
| 2021 | [Metro](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_End-to-End_Human_Pose_and_Mesh_Reconstruction_with_Transformers_CVPR_2021_paper.pdf) | Regression | - | transformer | mesh |  | x | ✓ |  |
| 2021 | [Hand4Whole](https://arxiv.org/pdf/2011.11534.pdf) | Regression | - | ResNet | params | ✓ | x | ✓ |  |
| 2021 | [PyMAF](https://arxiv.org/pdf/2103.16507.pdf) | Regression | - | CNN | params |  | x | x |  |
| 2021 | [PARE](https://arxiv.org/pdf/2104.08527.pdf) | Regression | - | CNN | params |  | x | x |  |
| 2021 | [Graphormer](https://arxiv.org/pdf/2104.00272.pdf) | Regression | - | transformer | mesh |  | x | x |  |
| 2021 | [TCMR](https://arxiv.org/pdf/2011.08627.pdf) | Regression | GRU | GRU | params |  | x | x | x |
| 2021 | [BMP](https://arxiv.org/pdf/2105.02467.pdf) | Regression | - | CNN. | mesh map |  | ✓ | x |  |
| 2020 | [TCN](https://arxiv.org/pdf/2004.11822.pdf) |  |  |  |  |  |  |  |  |
| 2020 | [MEVA](https://openaccess.thecvf.com/content/ACCV2020/papers/Luo_3D_Human_Motion_Estimation_via_Motion_Compression_and_Refinement_ACCV_2020_paper.pdf) | Regression | GRU | GRU | params |  | x | x |  |
| 2020 | [VIBE](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.pdf) | Regression | GAN | GAN | params |  | x | x |  |
| 2019 | [Bundle](https://openaccess.thecvf.com/content_CVPR_2019/papers/Arnab_Exploiting_Temporal_Context_for_3D_Human_Pose_Estimation_in_the_CVPR_2019_paper.pdf) | Optimization? | bundle adj | CNN | params |  | x | x | x |
| 2020 | [ROMP](https://arxiv.org/pdf/2008.12272.pdf) | Regression | - | CNN | params |  | ✓ | x |  |
| 2019 | [GraphCNN](https://arxiv.org/pdf/1905.03244.pdf) | Regression | - | Graph CNN | mesh | x |  |  |  |
| 2019 | [HMR](https://arxiv.org/pdf/1712.06584.pdf) | Regression | Image | x | x | x | ✓ |  |  |
| 2019 | [HMMR](https://arxiv.org/pdf/1812.01601.pdf) | Regression | Video | x | x | x | ✓ | semi |  |
| 2019 | [DenseRaC]() | Regression | - | CNN | IUV |  |  |  |  |
| 2019 | [SATN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sun_Human_Mesh_Recovery_From_Monocular_Images_via_a_Skeleton-Disentangled_Representation_ICCV_2019_paper.pdf) | Regression | Video |  |  |  |  |  |  |
| 2023 | [HybrIK-X](https://arxiv.org/abs/2304.05690) | Optimization | Image | ✓ | ✓ | ✓ | ✓ |  |  |
| 2022 | [t-HMMR](https://arxiv.org/pdf/2012.09843.pdf) | Optimization | transformer | transformer | params |  | x | x |  |
| 2021 | [HybrIK](https://arxiv.org/pdf/2011.14672.pdf) | Optimization | Image | ✓ | ✓ | ✓ | ✓ |  |  |
| 2021 | [HuMor](https://geometry.stanford.edu/projects/humor/docs/humor.pdf) | Optimization | Image | ✓ | ✓ | ✓ | ✓ |  |  |
| 2019 | [SMPLify-X](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.pdf) | Optimization | Image | ✓ | ✓ | ✓ | ✓ |  |  |
| 2019 | [SPIN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.pdf) | Optimization | Image | ✓ | ✓ | ✓ | ✓ | Fully |  |

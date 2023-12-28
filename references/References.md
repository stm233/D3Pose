
### Overview

#### Models

| Year | Model Name |Model-type| Temporal | Networks | output | interm | Multi-p* | Whole-body | Supervision|
| -----|------------|----------|-------   | ---      | ------ | -------| ---------| -----      | ----       |
|2023  | [POTTER](https://arxiv.org/pdf/2303.13357.pdf)    |Regression| -       | Vit      |
|2023  | [FeatER](https://arxiv.org/pdf/2205.15448.pdf)   | Regression | -      | Vit      |  
| 2023 | [Trace](https://arxiv.org/pdf/2306.02850.pdf)    | Regression| -       | 
| 2023 | [HMR 2.0](https://arxiv.org/pdf/2305.20091.pdf)  | Regression| -       | ViT      | params |       |      x    |        x  |       | 
| 2023 | [OSX](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.pdf)      | Regression |-       | ViT      | params |       | ✓         |
| 2023 | [SA-HMR](https://arxiv.org/pdf/2306.03847.pdf)   | Regression | -      | Trasformer| mesh   |       | x           | ✓        |
|2023 | [CoorFormer](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Coordinate_Transformer_Achieving_Single-stage_Multi-person_Mesh_Recovery_from_Videos_ICCV_2023_paper.pdf)| Regression | STT    | STT         | params map    |       |  ✓          |         |
|2022 | [FastMETRO](https://arxiv.org/pdf/2207.13820.pdf) | Regression | -      |Transformer         |mesh |       |  x |.  ✓  |   
|2022 | [CLIFF](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Learning_Analytical_Posterior_Probability_for_Human_Mesh_Recovery_CVPR_2023_paper.html)    | Regression | -      | CNN          | probability |   | x           | ✓        |
|2022 |[PyMAF-X](https://arxiv.org/pdf/2207.06400.pdf)   | Regression | -      | CNN          | params     |       | x    |x|
|2022 |[GLAMR](https://arxiv.org/pdf/2112.01524.pdf)     | Regression | | 
|2022 | [BEV](https://arxiv.org/pdf/2112.08274.pdf)       | Regression | -      | CNN         | mesh map | -   |✓ | x |         |
|2022| [OCHMR](https://arxiv.org/pdf/2203.13349.pdf)      | Regression | -      | CNN         |  params   |            | ✓        | x| 
|2022 | [MPS-Net](https://arxiv.org/abs/2203.08534)   | Regression | HAFI   | ResNet      | params |  | x| x| 
|2021 | [Metro](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_End-to-End_Human_Pose_and_Mesh_Reconstruction_with_Transformers_CVPR_2021_paper.pdf)     | Regression | -      | transformer | mesh |       | x      | ✓ | | 
|2021 | [Hand4Whole](https://arxiv.org/pdf/2011.11534.pdf) | Regression| -      | ResNet      | params  | ✓    | x        |  ✓ | 
|2021 | [PyMAF](https://arxiv.org/pdf/2103.16507.pdf)     | Regression | -      | CNN         | params  | |   x    | x        |
|2021 | [PARE](https://arxiv.org/pdf/2104.08527.pdf)      |Regression  | -      | CNN         | params    |   | x           | x       |
|2021 | [Graphormer](https://arxiv.org/pdf/2104.00272.pdf)|Regression  | -      | transformer | mesh    |   | x   |  x      |         |
|2021 | [TCMR](https://arxiv.org/pdf/2011.08627.pdf)|Regression| GRU            | GRU         | params  |  | x | x |x |
|2021 | [BMP](https://arxiv.org/pdf/2105.02467.pdf)| Regression         | -     | CNN.        | mesh map| | ✓ |  x|  
|2020 | [MEVA](https://openaccess.thecvf.com/content/ACCV2020/papers/Luo_3D_Human_Motion_Estimation_via_Motion_Compression_and_Refinement_ACCV_2020_paper.pdf)|Regression | GRU |                   GRU | params | | x | x |
|2020 | [VIBE](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.pdf)      |Regression|GAN  | GAN           | params     |    | x           | x        |
|2019 | [Bundle](https://openaccess.thecvf.com/content_CVPR_2019/papers/Arnab_Exploiting_Temporal_Context_for_3D_Human_Pose_Estimation_in_the_CVPR_2019_paper.pdf) | Optimization? | bundle adj|CNN | params |  |x | x| x|  
|2020 | [ROMP](https://arxiv.org/pdf/2008.12272.pdf)      |Regression|-  | CNN            |  params ||   ✓    | x       |
|2019 | [GraphCNN](https://arxiv.org/pdf/1905.03244.pdf)  |Regression  |-       | Graph CNN| mesh         | x    |  |  | | 
|2019 | [HMR](https://arxiv.org/pdf/1712.06584.pdf)       |Regression|Image | x           | x           | x           | ✓        |
|2019| [HMMR](https://arxiv.org/pdf/1812.01601.pdf)       |Regression|Video | x           | x           | x           | ✓        | semi|
|2019| [DenseRaC]()     |Regression| -| CNN| IUV | | ||  
|2019| [SATN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sun_Human_Mesh_Recovery_From_Monocular_Images_via_a_Skeleton-Disentangled_Representation_ICCV_2019_paper.pdf)       |Regression| Video|
|2023| [HybrIK-X](https://arxiv.org/abs/2304.05690)   | Optimization |  Image | ✓           | ✓           | ✓           | ✓        |
|2022 | [t-HMMR](https://arxiv.org/pdf/2012.09843.pdf)    |Optimization |transformer | transformer  | params |  | x           | x        |
|2021| [HybrIK](https://arxiv.org/pdf/2011.14672.pdf)     | Optimization |Image | ✓           | ✓           | ✓           | ✓        |
|2021| [HuMor](https://geometry.stanford.edu/projects/humor/docs/humor.pdf)      | Optimization | Image | ✓           | ✓           | ✓           | ✓        |
|2019| [SMPLify-X](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.pdf)  | Optimization | Image | ✓           | ✓           | ✓           | ✓        |
|2019| [SPIN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.pdf)       | Optimization|Image | ✓           | ✓           | ✓           | ✓        | Fully |


#### Datasets

| Year | Model Name |Datasets                                               |
| -----|------------   | ---                                                         |
| 2023 | [HMR 2.0](https://arxiv.org/pdf/2305.20091.pdf)  | Human3.6M, MPI-INF- 3DHP, COCO, MPII / InstaVariety, AVA, AI Challenger / 3DPW, PoseTrack|
| 2023 | [OSX](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.pdf)      | COCO-Wholebody, MPII, and Human3.6M / AGORA, EHF, 3DPW |
| 2023 | [SA-HMR](https://arxiv.org/pdf/2306.03847.pdf)   | RICH, PROX  |
|2023 | [CoorFormer](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Coordinate_Transformer_Achieving_Single-stage_Multi-person_Mesh_Recovery_from_Videos_ICCV_2023_paper.pdf)| MPI-INF-3DHP, MuCo-3DHP, MPII, LSP,3DPW, Human3.6M     |
| 2022 | [CLIFF](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Learning_Analytical_Posterior_Probability_for_Human_Mesh_Recovery_CVPR_2023_paper.html)    | Human3.6M, 3DPW, MS COCO, MPI-INF-3DHP, AGORA, TotalCapture        |
| 2022 |[PyMAF-X](https://arxiv.org/pdf/2207.06400.pdf)   | Human3.6M, MPI-INF-3DHP, MPII, LSP, LSP-extended, COCO, FreiHAND, InterHand2.6M, COCO-WholeBody, VGGFace2      |
|2022 | [BEV](https://arxiv.org/pdf/2112.08274.pdf)       | RH, Human3.6M, MuCo-3DHP, COCO, MPII, LSP, and CrowdPose, AGORA, ROMP      |
|2022| [OCHMR](https://arxiv.org/pdf/2203.13349.pdf)      | MPI-INF-3DHP, COCO, MPII, LSP-Extended     |
|2021 | [Hand4Whole](https://arxiv.org/pdf/2011.11534.pdf) | Human3.6M, MSCOCO, MPII, Frei-Hand, Stirling, EHF, AGORA, MSCOCO      |
|2021 | [PyMAF](https://arxiv.org/pdf/2103.16507.pdf)     |   |
|2021 | [PARE](https://arxiv.org/pdf/2104.08527.pdf)      |   COCO, MPII, LSPET, MPI-INF-3DHO, Human3.6M   |
|2021 | [Graphormer](https://arxiv.org/pdf/2104.00272.pdf)|  Human3.6M, MuCo-3DHP, UP-3D , COCO , MPII, 3DPW, FreiHand    |
|2020 | [VIBE](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.pdf)      | PennAction, PoseTrack, InstaVariety, Kinetics-400, MPI-INF-3DHP, Human3.6M, 3DPW, AMASS|
|2020 | [t-HMMR](https://arxiv.org/pdf/2012.09843.pdf)    |  Human3.6M, COCO, MPII, MS-AVA |
|2020 | [ROMP](https://arxiv.org/pdf/2008.12272.pdf)      | Human3.6M, MPI-INF-3DHP, UP, MS COCO, MPII, LSP, RICH, MuCo-3DHP, and OH, PoseTrack, Crowdpose, 3DPW |
|2019 | [HMR](https://arxiv.org/pdf/1712.06584.pdf)       | LSP, LSP-extended,  MPII, MS COCO   |
|2019| [HMMR](https://arxiv.org/pdf/1812.01601.pdf)       | Human3.6M, Penn Action, NBA dataset, 3DPW    |
|2023| [HybrIK-X](https://arxiv.org/abs/2304.05690)   |         |
|2021| [HybrIK](https://arxiv.org/pdf/2011.14672.pdf)     |        |
|2021| [HuMor](https://geometry.stanford.edu/projects/humor/docs/humor.pdf)      |        |
|2019| [SMPLify-X](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.pdf)  |       |
|2019| [SPIN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.pdf)       |      |


|Datasets |Annot| Env   | # Sample| # Subject| # Scenes| # Cam| Occ |
|---      | -   |-      |-        |-         |-        |-     |-    |
|[PROX](https://prox.is.tue.mpg.de)     | 2DKP |Indoor| 88484   |11        | 12      | -    |O    |
|COCO-Wholebody|  2DKP |Outdoor| 40055   |40055        | -      | -    |-    |
|[Instavariety](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md)| 2DKP |Outdoor| 2187158   |>28272        | -      | -    |-    |
|COCO| 2DKP |Outdoor| 28344   |28344        | -      | -    |-   |
|MuPoTs-3D| 2DKP | Outdoor| 20760   |8        |  -    | 12    |-    |
|LIP|2DKP |Outdoor| 25553   |25553        | -      | -    |-    |
|MPII|2DKP |Outdoor| 14810   |14810        | 3913      | -    |-    |
|Crowdpose|2DKP |Outdoor| 13927  |-        | -      | -    |P    |
|Vlog People|2DKP |Outdoor| 353306  |798        | 798      | -    |-    |
|PoseTrack| 2DKP |Outdoor| 5084   |550        | 550      | -    |-    |
|LSP |2DKP |Outdoor| 999  |999        | -      | -    |-    |
|[AI Challenger](https://arxiv.org/abs/1711.06475)|2DKP |Outdoor| 378374   |-        | -      | -    |-    |
|LSPET|2DKP |Outdoor| 9427   |9427        | -      | -    |-    |
|Penn-Action|2DKP |Outdoor| 17443  |2326        | 2326      | -    |-    |
|OCHuman (OCH)|2DKP |Outdoor| 10375   |8110        | -      | -    |P,O    |
|MuCo-3DHP (MuCo)|2DKP |Indoor| 482725   |8        | -      | 14    |-    |
|MPI-INF-3DHP (MI)| 2DKP/3DKP |Indoor| 105274  |8        | 1      | 14    |-    |
|3DOH50K (OH) |2DKP/3DKP |Indoor| 50310   |-        | 1      | 6    |O    |
|3D People|2DKP/3DKP |Indoor| 1984640 |80        | -      | 4    |-    |
|AGORA|2DKP/3DKP/SMPL |Indoor| 100015   |>350        | -      | -    |P,O    |
|SURREAL|2DKP/3DKP/SMPL |Indoor| 1605030   |145        | 2607      | -    |-    |
|Human3.6M (H36M)|2DKP/3DKP/SMPL |Indoor| 312188   |9        | 1      | 4    |-    |
|EFT-COCO |2DKP/SMPL |Outdoor| 74834   |74834        | -      | -    |-    |
|EFT-COCO-part|2DKP/SMPL |Outdoor| 28062  |28064        | -      | -    |-    |
|EFT-PoseTrack|2DKP/SMPL |Outdoor| 28457   |550        | -      | -    |-    |
|EFT-MPII|2DKP/SMPL |Outdoor| 14667  |3913        | -      | -    |-    |
|UP-3D|2DKP/SMPL |Outdoor| 7126   |7126        | -      | -    |-    |
|MTP|2DKP/SMPL |Outdoor| 3187   |3187        | -      | -    |-    |
|EFT-OCHUMAN|2DKP/SMPL |Outdoor| 2495   |2495        | -      | -    |P,O    |
|EFT-LSPET|2DKP/SMPL |Outdoor| 2946  |2946        | -      | -    |-    |
|[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)|SMPL |Outdoor| 22735  |7        | -      | -    |-    |

#### Loss Functions

| Year | Model Name |Eval Metrics                   | Loss Function |
| -----|------------   | ---                     | ---- | 
| 2023 | [HMR 2.0](https://arxiv.org/pdf/2305.20091.pdf)  | MPJPE, PA-MPJPE, PCK       | Lsmpl, Lkp3D, Lkpt2D, Ladv |
| 2023 | [OSX](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.pdf)      | MPJPE, PA-MPVPE, N-PMVPE | Lsmplx + Lkpt3D + Lkpt2D + Lbbox2D 
| 2023 | [SA-HMR](https://arxiv.org/pdf/2306.03847.pdf)   | G-MPJPE, G-MPVE, MPJPE and MPVE, PenE and ConFE  |  human vertices, human joints, recon- structed contact points, and global human vertices,        |
|2023 | [CoorFormer](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Coordinate_Transformer_Achieving_Single-stage_Multi-person_Mesh_Recovery_from_Videos_ICCV_2023_paper.pdf)| MPJPE, PAMPJPE, PVE| temporal + spatial        |
| 2022 | [CLIFF](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Learning_Analytical_Posterior_Probability_for_Human_Mesh_Recovery_CVPR_2023_paper.html)    | MPJPE, PA-MPJPE, and PVE  | ✓        |
| 2022 |[PyMAF-X](https://arxiv.org/pdf/2207.06400.pdf)   | PVE, MPJPE, PA-PVE, and PA-MPJPE|        |
|2022 | [BEV](https://arxiv.org/pdf/2112.08274.pdf)       | PCDR, mPCK, MPJPE, MVE| ✓        |
|2022| [OCHMR](https://arxiv.org/pdf/2203.13349.pdf)      | MPJPE, PMPJPE, PVE, AP, AR  | ✓        |
|2021 | [Hand4Whole](https://arxiv.org/pdf/2011.11534.pdf) | MPJPE, MPVPE, PA MPJPE, PA MPVPE       | ✓        |
|2021 | [PyMAF](https://arxiv.org/pdf/2103.16507.pdf)     |  | ✓        |
|2021 | [PARE](https://arxiv.org/pdf/2104.08527.pdf)      | MPJPE, PAMPJPE, PVE   | ✓        |
|2021 | [Graphormer](https://arxiv.org/pdf/2104.00272.pdf)| MPVE, MPJPE, PA-MPJPE | ✓        |
|2020 | [VIBE](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.pdf)      |  PVE, MPJPE, PA-MPJPE   | ✓        |
|2020 | [t-HMMR](https://arxiv.org/pdf/2012.09843.pdf)    |  PCK    | ✓        |
|2020 | [ROMP](https://arxiv.org/pdf/2008.12272.pdf)      |  PVE, MPJPE, PMPJPE. PCK, AUC, MPJAE, AP   | ✓        |
|2019 | [HMR](https://arxiv.org/pdf/1712.06584.pdf)       |  MPJPE, PCK, AUC     | ✓        |
|2019| [HMMR](https://arxiv.org/pdf/1812.01601.pdf)       |  MPJPE, PA-MPJPE, PCK      |  L2d, L3d, Ladv        |
|2023| [HybrIK-X](https://arxiv.org/abs/2304.05690)   |    | ✓        |
|2021| [HybrIK](https://arxiv.org/pdf/2011.14672.pdf)     |       | ✓        |
|2021| [HuMor](https://geometry.stanford.edu/projects/humor/docs/humor.pdf)      |         | ✓        |
|2019| [SMPLify-X](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.pdf)  |         | ✓        |
|2019| [SPIN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.pdf)       |     | ✓        |


#### Survey

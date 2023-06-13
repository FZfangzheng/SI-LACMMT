# SI-LACMMT

Level-Aware Consistent Multi-level Map Translation From Satellite Imagery  

[Ying Fu](https://ying-fu.github.io/), Zheng Fang, Linwei Chen, Tao Song, and Defu Lin 

<img src="https://github.com/FZfangzheng/SI-LACMMT/blob/master/img/example1.png" alt="图片替换文本" width="350" align="bottom" />

## 1. Image2image translation and map generation methods

* [x] [Pix2pix](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html) (CVPR 2017)
* [x] [Pix2pixHD](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.html) (CVPR 2018)
* [x] [CycleGAN](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html) (ICCV 2017)
* [x] [SPADE](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html) (CVPR 2019)
* [x] [SelectionGAN](https://openaccess.thecvf.com/content_CVPR_2019/html/Tang_Multi-Channel_Attention_Selection_GAN_With_Cascaded_Semantic_Guidance_for_Cross-View_CVPR_2019_paper.html) (CVPR 2019)
* [x] [TSIT](https://link.springer.com/chapter/10.1007/978-3-030-58580-8_13) (ECCV 2020)
* [x] [LPTN](https://openaccess.thecvf.com/content/CVPR2021/html/Lin_Drafting_and_Revision_Laplacian_Pyramid_Network_for_Fast_High-Quality_Artistic_CVPR_2021_paper.html?ref=https://githubhelp.com) (CVPR 2021)
* [x] [SMAPGAN](https://ieeexplore.ieee.org/document/9200723/) (TGRS 2020)
* [x] [CreativeGAN](https://ieeexplore.ieee.org/document/9540226) (TGRS 2021)
* [x] [SI-LACMMT](https://ieeexplore.ieee.org/document/9950295/) (TGRS 2022)
* [x] [LAMG_MLMG](https://ieeexplore.ieee.org/document/9764398) (JSTAR 2022) 
* [x] [SingleLevelMapGenerator](http://www.cjig.cn/jig/ch/reader/view_abstract.aspx?file_no=202208300000001) (中国图象图形学报 2023) 

| Method       | FID$\downarrow$ | KMMD$\downarrow$ | WD$\downarrow$ | PSNR$\uparrow$ | Model          |
| ------------ | --------------- | ---------------- | -------------- | -------------- | -------------- |
| Pix2pix      | 342.84          | 0.53236          | 16.267         | 20.726         | [Baidu Disk]() |
| Pix2pixHD    | 331.10          | 0.38806          | 14.358         | 20.908         | [Baidu Disk]() |
| CycleGAN     | 312.14          | 0.45603          | 14.692         | 20.725         | [Baidu Disk]() |
| SPADE        | 459.11          | 0.79792          | 20.047         | 20.468         | [Baidu Disk]() |
| SelectionGAN | 337.83          | 0.58850          | 16.475         | 20.617         | [Baidu Disk]() |
| TSIT         | 284.17          | 0.43861          | 13.753         | 20.540         | [Baidu Disk]() |
| LPTN         | 351.61          | 0.4380           | 17.489         | 21.327         | [Baidu Disk]() |
| SMAPGAN      | 336.35          | 0.54771          | 16.338         | $\bf{22.506}$  | [Baidu Disk]() |
| CreativeGAN  | 267.37          | 0.29931          | 12.453         | 21.428         | [Baidu Disk]() |
| SI-LACMMT    | $\bf{195.64}$   | $\bf{0.23759}$   | $\bf{11.014}$  | 21.532         | [Baidu Disk]() |
| LAMG_MLMG    | -               | -                | -              | -              | [Baidu Disk]() |

## 2. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages:

```
pip install -r requirements.txt
```

## 3. [MLMG Dataset](https://github.com/FZfangzheng/MLMG):

The advantages of our dataset over other datasets are as follows：

![image-20221214203605415](https://github.com/FZfangzheng/SI-LACMMT/blob/master/img/data1.png)

Our data contains multiple types of images, and has multi-level data in the same area. In addition, we have balanced the data of each level to ensure the balance of training.

![image-20221214203621871](https://github.com/FZfangzheng/SI-LACMMT/blob/master/img/data2.png)

The specific content of our data set is as shown above, including data from US and CN countries. The data volume of each level of training set is 2k. The test set is multi-level data in the same region, and the number of high-level data is four times that of low-level data.

following form:

```shell
|--MLMG Dataset
    |--US_dataset
    	|-- trainA(SI images)
    		|-- 15
    			|-- 15_26967_12413.jpg
    			|-- 15_26967_12415.jpg
    			：
                |-- 15_27091_12529.jpg
    		|-- 16
    		|-- 17
    		|-- 18
    	|-- trainB(Map images)
    	|-- trainC(2x downsample map images )
    	|-- train_seg(element labels)
    	|-- testA(SI images)
    	|-- testB(Map images)
    |--CN_dataset
    	|-- trainA
    	|-- trainB
    	|-- trainC
    	|-- train_seg
    	|-- testA
    	|-- testB
```

## 4. Experiement:

### 4.1　Training

1. Download the dataset and move it to the dataset folder.
2. Download the [pretrained model](https://pan.baidu.com/s/1i4wnqdI1iYmAwImzdTVFpA?pwd=up35) and move it to src/LACMMT/.
3. Execute the following command to train the corresponding model.
```shell
cd "root_path"
bash scripts/train_CreativeGAN.sh
bash scripts/train_LACMMT.sh
bash scripts/train_LAMG_MLMG.sh
```

### 4.2　Testing	

1. Download the test dataset and the pretrained model provided above.
2. Configure the python environment according to the requests.txt.
3. Download the dataset and move it to the dataset folder.
4. Execute the following script to generate the multi-level map images.

```shell
cd "root_path"
bash scripts/test_LACMMT.sh
bash scripts/test_Pix2pix.sh
:
bash scripts/test_CreativeGAN.sh
```

## 5.Citation

```bibtex
@article{fu2022level,
  title={Level-Aware Consistent Multilevel Map Translation From Satellite Imagery},
  author={Fu, Ying and Fang, Zheng and Chen, Linwei and Song, Tao and Lin, Defu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--14},
  year={2022},
  publisher={IEEE}
}
@article{chen2022consistency,
  title={Consistency-Aware Map Generation at Multiple Zoom Levels Using Aerial Image},
  author={Chen, Linwei and Fang, Zheng and Fu, Ying},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={15},
  pages={5953--5966},
  year={2022},
  publisher={IEEE}
}
@article{FZ_transformer,
  title={Transformer特征引导的双阶段地图智能生成},
  author={方政 and 付莹 and 刘利雄}
}
```

## 6.News :sparkles:

- 2023-1-1: release train code for CreativeGAN, LAMG_MLMG, and LACMMT.
- 2022-12-18: release test code.
- 2022-8-30: release dataset.
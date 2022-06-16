# [Joint Forecasting of Panoptic Segmentations with Difference Attention](https://arxiv.org/abs/2204.07157)
**Colin Graber, Cyril Jazra, Wenjie Luo, Liangyan Gui, Alexander Schwing - CVPR 2022**

## ⚙️ Setup

### Dependencies
- pytorch v.1.10.0 
- detectron2 version 0.4.1 (newer versions will cause errors)
- cityscapesscripts
- opencv
- PIL 
- numpy

Install the code using the following command:
`pip install -e ./`

### Data 
You will need to download the following datasets from the Cityscapes website. They should be placed in the data/cityscapes/ folder:
- gtFine_trainvaltest.zip
- leftImg8bit_sequence_trainvaltest.zip
- vehicle_sequence.zip
- timestamp_sequence.zip


## Running our code
use the script run.sh to train the model
There are three steps that can be run in the current version of this code:
1) extracting the appearance features for every instance using the pre-trained model.
2) training the box forecasting model
3) training the appearance forecasting model

COMING SOON: the steps required to train the refinement head. This includes a large amount of data preprocessing, and this is coming soon!

## ✏️ 📄 Citation

If you found our work relevant to yours, please consider citing our paper:
```
@inproceedings{graber-2022-panopticforecasting,
 title   = {Joint Forecasting of Panoptic Segmentations with Difference Attention},
 author  = {Colin Graber and
            Cyril Jazra and
            Wenjie Luo and
            Liangyan Gui and
            Alexander Schwing},
 booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
 year = {2022}
}
```

# License
This code is distributed under the MIT License. See `LICENSE.md` for details.
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

Training the prediction refinement head includes a large amount of data preprocessing, and an attempt at packaging this code in a manner that can be run by other people is underway. The model code is included and can be read through.

## Model Predictions
Due to the large amount of compute time and disk space required to prepare data for the prediction refinement head to run, it will not be practical for most people. To facilitate comparisons between our approach, we have included the final model panoptic segmentation outputs for the short- and mid-term settings on Cityscapes in the `predictions/` folder.

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

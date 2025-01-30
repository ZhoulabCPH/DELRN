# DELRN
****
## Dataset
- CHCAMS, Chinese Academy of Medical Sciences.
- SCCH, Sichuan Cancer Hospital.
- TMUCIH, Tianjin Medical University Cancer Institute & Hospital.
- SHCH, Shanghai Chest Hospital

## Predictive models training

### Data preprocessing


Resampling the CT images to the specified voxel dimensions.

```
python ./_1_data_processing/img_preprocessing_resample.py 
```

Obtaining the tumor's bounding box based on the IMG and MASK after resampling.

```
python ./_1_data_processing/get_bounding_box_mt.py 
```

#### Deep feature extractor training

Training the deep feature extractor.

```
python ./_2_deep_model_train/train.py 
```

Evaluating the deep feature extractor.

```
python ./_2_deep_model_train/eval.py 
```

#### Deep feature extracting

Extracting deep learning features of the tumor from bounding boxes.

```
python ./_3_deep_feature_extract/deep_feat_extract.py 
```

#### Radiomic feature extracting

Extracting radiomic features of the tumor from ROIs.

```
python ./_4_radio_feature_extract/radio_feat_extract.py 
```

#### Model Constrution

Training the DELRN model leveraging radiomic features and deep featuresd

```
python ./_5_feature_fusion/train.py 
```









  





  

# BdSL40 Dataset
This Bangla Sign Language Dataset **BdSL40** comprises of 611 videos over 40 BdSL words with 8 to 22 video clips per word. 

## AI for Bangla 2.0
This project received an Honorable Mention at the [AI for Bangla 2.0](https://bangla.gov.bd/ai-for-bangla-2-0/) Competition, under team name -- *Silent Sentinels*.

## Dataset Link
[BdSL40 Video](https://drive.google.com/file/d/1mSId206Y5enSRsW812Ike_-eP94ZG9Kl/view?usp=sharing) <br>
[BdSL40 Frames](https://www.kaggle.com/datasets/sameen53/bdsl-images)

## ISL to BDSL Mapping
[ISL-to-BDSL](https://docs.google.com/document/d/1MfRud55u45gtoPpB6GLAp18vrfpgD2ek3mO0qUJhv7s/edit?usp=sharing). This mapping was created according to [BdSL Dictionary](https://www.scribd.com/doc/251910320/Bangla-Sign-Language-Dictionary) and [ISL Dataset](https://zenodo.org/records/4010759). It is relatively easy to reproduce BdSL40 from ISL by following the mapping.

## Kaggle Notebook
[Training_on_BdSL40](https://www.kaggle.com/sameen53/bdsl40-3dcnn-githubv2)


## Dataset Properties
| Criterion             | Count       |
| ----------------------| ----------- |
| Videos                | 611         |
| Words                 | 40          |


| Label           |  Count   || Label      |  Count   |
| ----------------| -------- |-| ----------------| -------- |
| new             | 21       || bad             | 21       |
| lawyer          | 14       || teacher         | 14       |
| yesterday       | 14       || time            | 15       |
| friend          | 20       || i               | 21       |
| you             | 21       || telephone       | 14       |
| ring            | 14       || winter          | 14       |
| brown           | 21       || skirt           | 19       |
| pant            | 20       || shoes           | 20       |
| camera          | 14       || heavy           | 8        |
| soap            | 14       || book            | 14       |
| india           | 21       || quiet           | 21       |
| deaf            | 8        || rich            | 8        |
| thick           | 8        || money           | 14       |
| cow             | 21       || fulfill         | 8        |
| cheap           | 8        || straight        | 8        |
| life            | 8        || more            | 21       |
| crane           | 20       || shirt           | 20       |
| noon            | 14       || bed             | 14       |
| square          | 14       || glad            | 8        |
| tortoise        | 20       || student         | 14       |

## Samples
**Label:** Student

![Alt_test](https://github.com/PatchworkProgrammer/BdSL40_Dataset/blob/main/Resources/student.gif)


**Label:** Tortoise

![Alt_test](https://github.com/PatchworkProgrammer/BdSL40_Dataset/blob/main/Resources/tortoise.gif)

## Preprocessing
VideoResnet requires videos to be sampled into images and cropped to a square ratio. 32 frames are sampled from each video and the height and width are set to 100 pixels by default. 

Usage:

    python Preprocessing/preprocessing.py [path_to_dataset] [path_to_save_images]

## Hyperparameters

The following hyperparameters are set by default.

    "num_epochs": 120,
    "learning_rate": 5e-5,
    "batch_size": 64,
    "h": 100,
    "w": 100,
    "mean": [0.5, 0.5, 0.5],
    "std":  [0.5, 0.5, 0.5],
    "total_frames": 32,
    "start_skip": 6,
    "end_skip": 8,
    "batch_size": 64,
    "num_classes": 40,
    "test_ratio": 0.2
  
## Training

Usage:

    python Training/training.py [path_to_save_images]
    
## Results after Training
Accuracy 82.93%

![Result2](https://github.com/PatchworkProgrammer/BdSL40_Dataset/blob/main/Resources/results.png)


## Acknowledgements

This dataset is adapted from by [INCLUDE](https://zenodo.org/record/4010759) by [Sridhar et al](https://doi.org/10.1145/3394171.3413528) under the [Creative Commons 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) license.

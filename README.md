# BdSL40_Dataset
Bangla Sign Language Dataset (BdSL40) comprises of 611 videos over 40 BdSL words with 8 to 22 video clips per word. 

##Dataset properties
| Criterion             | Count       |
| ----------------------| ----------- |
| Videos                | 611         |
| Words                 | 40          |




| Word/Label      |  Count   |
| ----------------| -------- |
| new             | 21       |
| bad             | 21       |
| lawyer          | 14       |
| teacher         | 14       |
| yesterday       | 14       |
| time            | 15       |
| friend          | 20       |
| i               | 21       |
| you             | 21       |
| telephone       | 14       |
| ring            | 14       |
| winter          | 14       |
| brown           | 21       |
| skirt           | 19       |
| pant            | 20       |
| shoes           | 20       |
| camera          | 14       |
| heavy           | 8        |
| soap            | 14       |
| book            | 14       |
| india           | 21       |
| quiet           | 21       |
| deaf            | 8        |
| rich            | 8        |
| thick           | 8        |
| money           | 14       |
| cow             | 21       |
| fulfill         | 8        |
| cheap           | 8        |
| straight        | 8        |
| life            | 8        |
| more            | 21       |
| crane           | 20       |
| shirt           | 20       |
| noon            | 14       |
| bed             | 14       |
| square          | 14       |
| glad            | 8        |
| tortoise        | 20       |
| student         | 14       |


##Samples


## Preprocessing
VideoResnet requires videos to be sampled into images and cropped to a square ratio. 32 frames are sampled from each video and the height and width are set to 100 pixels by default.

    python preprocessing.py [path_to_dataset] [path_to_save_images]

##Hyperparameters

The following hyper_parameters are set by default.

    "num_epochs": 300,
    "learning_rate": 5e-5,
    "batch_size": 64,
    "h": 100,
    "w": 100,
    "mean": [0.43216, 0.394666, 0.37645],
    "std":  [0.22803, 0.22145, 0.216989],
    "total_frames": 32,
    "start_skip": 8,
    "end_skip": 8,
    "batch_size": 64,
    "num_classes": 40,
    "test_ratio": 0.2
  
## Training
    python training.py [path_to_save_images]
    

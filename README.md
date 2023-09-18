# VietnameseGAN (TensorFlow 2/ Python3)

Implementation of [ScrabbleGAN](https://arxiv.org/pdf/2003.10557.pdf) for adversarial generation of handwritten text images in TensorFlow 2.1.

This repo has been adjusted and changed for Vietnamese handwritting characters generation. It is used for the purpose of generating synthetic training data for our model in OCR task - AI4ALL 2023 competetion. For more information, please refer this: [AI4ALL](https://bkai.ai/soict-hackathon-2023/?fbclid=IwAR14-5SGJdQmSVWU2tQlbkNCgdBx46LZvZQyjgJugop5k2wMdonWLBTA-Ng)

<p align="center">
  <img src="doc\biggan.gif" />
</p>

 ## Try it out!
 [<img src="https://seeklogo.com/images/K/kaggle-logo-83322F52DE-seeklogo.com.png" align="center">](https://www.kaggle.com/code/nmddfdfd/scrabblegan-gen/notebook)

 Example on how to train ScrabbleGAN + how to generate images. [More details can be found here](https://towardsdatascience.com/scrabblegan-adversarial-generation-of-handwritten-text-images-628f8edcfeed).

 ## IAM database reformating (Optional)

 ### NOTE: You need to make sure the data are in the IAM DB format and stored in res/data/<database_name>d. Unless this condition is satisfied, the implementation will be failed.

 - We reconstruct the format of original dataset and rename it to IAM DB format. The detailed implementation can be found in `convert2IAM.py`
- For converting, run the following command:
```bash
python convert2IAM.py --train-list <path/to/train/txt> --val-list <path/to/val/txt> --images-dir <path/to/image/dir>
```

 ## Training and weights

 - We train with public dataset from AI4ALL OCR track dataset, including about 10k Vietnames handwriting images.
 - We formulated the dataset as IAM-Handwriting DB format, and followed the original instructions
 - We trained for 15 epochs due to resource limitation. The trained weights can be found [here](https://www.kaggle.com/datasets/nmddfdfd/scrabble-gan-modelv1)

```bash
python src/main.py
```
 
 ## To generate Vietnamese HW
 ```bash
 python src/generate.py --weight <path/to/trained/TFModel> --text-path <path/to/txt/list>
 ```
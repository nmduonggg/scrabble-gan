# VietnameseGAN (TensorFlow 2/ Python3)

Implementation of [ScrabbleGAN](https://arxiv.org/pdf/2003.10557.pdf) for adversarial generation of handwritten text images in TensorFlow 2.1.

This repo has been adjusted and changed for Vietnamese handwritting characters generation. It is used for the purpose of generating synthetic training data for our model in OCR task - AI4ALL 2023 competetion. For more information, please refer this: [AI4ALL](https://bkai.ai/soict-hackathon-2023/?fbclid=IwAR14-5SGJdQmSVWU2tQlbkNCgdBx46LZvZQyjgJugop5k2wMdonWLBTA-Ng)

<p align="center">
  <img src="doc\biggan.gif" />
</p>

 ## Try it out!
 [<img src="https://seeklogo.com/images/K/kaggle-logo-83322F52DE-seeklogo.com.png" align="center">](https://www.kaggle.com/code/nmddfdfd/scrabblegan-gen/notebook)

 Example on how to train ScrabbleGAN + how to generate images. [More details can be found here](https://towardsdatascience.com/scrabblegan-adversarial-generation-of-handwritten-text-images-628f8edcfeed).

 ## Training and weights

 - We train with public dataset from AI4ALL OCR track dataset, including about 10k Vietnames handwriting images.
 - We formulated the dataset as IAM-Handwriting DB format, and followed the original instructions
 - We trained for 15 epochs due to resource limitation. The trained weights can be found [here](https://www.kaggle.com/datasets/nmddfdfd/scrabble-gan-modelv1)

 ## To generate Vietnamese HW
 ```
 python src/generate.py --saved-model <path/to/trained/TFModel>
 ```
    
 # Original paper: ScrabbleGan    
 ## Setup (Original paper)
 
 1. Download and extract [IAM-Handwriting DB](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database) 
 and [random word list](https://www.dropbox.com/s/o60ubzh3m163j0x/random_words.txt?dl=0) as described in src/dinterface/dinterface.py
    
    ```
    wget --user <user-id> --password <pw> http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz 
    wget --user <user-id> --password <pw> http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt
    wget -O random_words.txt https://www.dropbox.com/s/o60ubzh3m163j0x/random_words.txt?dl=0
    tar zxvf words.tgz
    rm -rf words.tgz       
    ```      
 2. Move folders into correct position (see File Structure); update file paths in scrabble_gan.gin + main.py 
 3. Install dependencies (e.g. [AWS P3](doc/installation_aws_p3.txt))
 
 ## Usage
  
Train model
    
    python3 main.py
 
 ## File Structure
 
     doc
         ├── installation_aws_p3.txt                    # how to install dependencies on AWS P3 instance
         ├── ...                                        # addtional README resources
     res                               
         ├── data/                                      # various word datasets (e.g. RIMES, IAM, ...)
             ├── iamDB/                                              
                 ├── words/                                          
                 ├── words-Reading/ 
                 ├── words.txt                                            
             ├── random_words.txt/                                  
         ├── out/                                       # location where to store generated output files   
             ├── big_ac_gan/                            # output of ScrabbleGAN (here denoted as big_ac_gan)
                 ├── images/                            
                 ├── model/                             
                 ├── training_checkpoints/              
             ├── ...                                    # potentially output of more models   
     src
         ├── bigacgan/                                  # ScrabbleGAN implementation
             ├── ...                                           
         ├── dinterface/                                # purpose: bring various datasets into equal format
             ├── ...                                         
         ├── main.py                                    # core of this repo - train ScrabbleGAN
         ├── scrabble_gan.gin                           # all dependencies are handled here (gin config file)
    
 ## License
 [License](LICENSE)

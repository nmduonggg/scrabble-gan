import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import cv2

import argparse

def get_parse():
    args = argparse.ArgumentParser()
    args.add_argument(
        '--text-path',
        help='Path to text list that you want to generate',
        default='res/data/viet74k.txt'
    )
    args.add_argument(
        '--weight',
        help='Path to trained model',
        required=True
    )
    args.add_argument(
        '--output-dir',
        help='Directory to save ouput: images and annotations',
        default='outputs/'
    )
    args.add_argument(
        '--num-unqiues',
        type=int,
        help='Number of unique texts taken from list',
        default=10000
    )
    args.add_argument(
        '--variation',
        help='Number of different style for each text sample',
        default=3
    ),
    args.add_argument(
        '--show',
        help='Show periodically or not (used for notebook)',
        action='store_true'
    )
    
    return args.parse_args()

args = get_parse()

# Setup input
text_paths = args.text_path
texts = []
invalid_chars = [',', '.', '?', '[', ']', '!', '#', '$', ':', ';']
with open(text_paths, 'r') as f:
    for line in f.readlines():
        line = line.translate({ord(x): '' for x in invalid_chars})
        texts.append(line.strip())
        
# Setup model
latent_dim = 128
char_vec = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ- '
path_to_saved_model = args.weight
imported_model = tf.saved_model.load(path_to_saved_model)

# Setup outptut
save_annot_dir = os.path.join(args.output_dir, "annots")
save_image_dir = os.path.join(args.output_dir, "images")

if not os.path.exists(save_annot_dir):
    os.makedirs(save_annot_dir)
if not os.path.exists(save_image_dir):
    os.makedirs(save_image_dir)
    
save_annot_path = os.path.join(save_annot_dir, "annotations.txt")

# Setup functions
def write_annot(image_name, label, img_id):
    if img_id == 0:
        with open(save_annot_path, 'w') as f:
            f.write(f"{image_name} {label}\n")
    else:
        with open(save_annot_path, 'a') as f:
            f.write(f"{image_name} {label}\n")
            

def show_or_save(text: str, n_samples: int, save=False, show=True):
    global img_id
    for idx in range(1):
        fake_labels = []
        words = [text] * n_samples
        noise = tf.random.normal([n_samples, latent_dim])
        
        for word in words:
            fake_labels.append([char_vec.index(char) for char in word])
        fake_labels = np.array(fake_labels, np.int32)

      # run inference process
        predictions = imported_model([noise, fake_labels], training=False)
      # transform values into range [0, 1]
        predictions = (predictions + 1) / 2.0

      # plot results
    for i in range(predictions.shape[0]):
        plt.subplot(10, 1, i + 1)
        pred = predictions[i, :, :, 0]
        pred = np.squeeze(pred)
        stacked_img = (np.stack((pred,)*3, axis=-1) * 255).astype(np.int32)
        if show:
            plt.imshow(stacked_img)
            plt.text(0, -1, "".join([char_vec[label] for label in fake_labels[i]]))
            plt.axis('off')
            plt.show()
        if save:
            image_path = f"images/gen_image_{img_id}.jpg"
            name = os.path.basename(image_path)
            cv2.imwrite(image_path, stacked_img)
            write_annot(name, text.replace(" ", ""), img_id)
            img_id += 1
            
img_id = 0

def main():
    num_uniqnues = args.num_uniques
    variation = args.variation
    for i, text in tqdm.tqdm(enumerate(texts[:num_uniqnues]), total=num_uniqnues):
        try:
            if i % 100 == 0:
                show_or_save(text, variation, save=True, show=True)
            else:
                show_or_save(text, variation, save=True, show=False)
        except: 
            log_info = f"[ERROR] Exception on {text}"
            print(log_info)
import os
import argparse
import tqdm
import cv2

def get_parse():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-tl",
        "--train-list",
        help="Train text list path",
        required=True
    )
    args.add_argument(
        "-vl",
        "--val-list",
        help="Valid text list path",
        required=True
    )
    args.add_argument(
        '--images',
        help="Image directory",
        required=True
    )
    args.add_argument(
        '-out',
        '--out-dir',
        default="./res/data/vnDB"
    )
    return args.parse_args()


args = get_parse()
save_img_dir = os.path.join(args.out_dir, "words")
txt_file = os.path.join(args.out_dir, "words.txt")

if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)

paths = [args.train_list, args.val_list]
for i,path in enumerate(paths):
    with open(path, 'r') as f1:
        for j, line in tqdm.tqdm(enumerate(f1.readlines())):
            if (i==0 and j==0):
                with open(txt_file, 'w') as f2:
                    img_name, label = line.split()
                    f2.write(f"{img_name} ok {line}")
            else:
                with open(txt_file, 'a') as f2:
                    img_name, label = line.split()
                    f2.write(f"\n{img_name} ok {line}")
                    
for img_name in tqdm.tqdm(os.listdir(args.images_dir)):
    try:
        image = cv2.imread(os.path.join(args.images_dir, img_name))
        cv2.imwrite(os.path.join(save_img_dir, img_name), image)
    except:
        print(f"Error on {img_name}")
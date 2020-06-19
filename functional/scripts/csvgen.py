import os, csv

path = '/datasets/zlai/train_all_frames/JPEGImages'
ld = os.listdir(path)

with open('ytvos_train.csv', 'w') as f:
    filewriter = csv.writer(f)
    for l in ld:
        n = len(os.listdir(os.path.join(path,l)))
        filewriter.writerow([l, n])

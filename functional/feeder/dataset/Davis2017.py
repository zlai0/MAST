import os

catnames = None
def dataloader(filepath):
    global catnames
    catname_txt = filepath + '/ImageSets/2017/val.txt'

    catnames = open(catname_txt).readlines()

    annotation_all = []
    jpeg_all = []

    for catname in catnames:
        anno_path = os.path.join(filepath, 'Annotations/480p/' + catname.strip())
        cat_annos = [os.path.join(anno_path,file) for file in sorted(os.listdir(anno_path))]
        annotation_all.append(cat_annos)

        jpeg_path = os.path.join(filepath, 'JPEGImages/480p/' + catname.strip())
        cat_jpegs = [os.path.join(jpeg_path, file) for file in sorted(os.listdir(jpeg_path))]
        jpeg_all.append(cat_jpegs)

    return annotation_all, jpeg_all
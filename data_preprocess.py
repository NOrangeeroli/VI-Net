import json,os
import _pickle as cPickle
from collections import defaultdict




def load_stats(data_dir, dataset,list_name):
    img_path = os.path.join( dataset,list_name )
    print(img_path)
    img_list = [os.path.join(data_dir,img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(data_dir, img_path))]
    dict_cat = defaultdict(list)
    for img in img_list:
        with open(img + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
            for instance,cls  in enumerate(gts['class_ids']):
                dict_cat[cls].append((img, instance))
    
    return dict_cat


data_dir = "/media/student/Data/yamei/data/NOCS/"

camera_train_stats = load_stats(data_dir, 'camera', 'train_list.txt')
with open(os.path.join(data_dir, 'camera', 'train_category_dict.json'), 'w') as fp:
    json.dump(camera_train_stats, fp)
real_train_stats = load_stats(data_dir, 'real', 'train_list.txt')

with open(os.path.join(data_dir, 'real', 'train_category_dict.json'), 'w') as fp:
    json.dump(real_train_stats, fp)



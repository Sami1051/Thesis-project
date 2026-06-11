import tensorflow as tf
import numpy as np
import cv2
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"



db='test_ugrad_noaug_500px'
#db='train_ugrad_noaug_500px_clutter'
#db='train_ugrad_noaug_500px_mines'
#db='train_ugrad_noaug_500px'



read_index=False
v_2024=False
data_set_type='Train Data'
read_quality=False
class_filtering=None

if (db=='train_ugrad_noaug_500px_clutter'):
    image_list_file='/home/ndw/ugrad_data/train_500px/png_snippets_q.txt'
    image_folder_in='/home/ndw/ugrad_data/train_500px/'
    image_folder_out='/home/ndw/ugrad_data/tf_data/dataset_2/'
    tfrecord_file_base ='clutter_noaug_500px_train_i'
    tfrecord_len= 3000
    performed = '2025_11_10'
    data_set_type='Train Data'
    read_quality=True
    class_filtering='filter'
    classes_to_filter=[26, 49,51]
    comment='ugrad data exp 1'

if (db=='train_ugrad_noaug_500px_mines'):
    image_list_file = '/home/ndw/ugrad_data/train_500px/png_snippets_q.txt'
    image_folder_in = '/home/ndw/ugrad_data/train_500px/'
    image_folder_out = '/home/ndw/ugrad_data/tf_data/dataset_2/'
    tfrecord_file_base ='mines_s_noaug_500px_train_i'
    tfrecord_len= 37 # 3000
    performed = '2026_02_02' #''2025_11_10'
    data_set_type='Train Data'
    read_quality=True
    class_filtering='retain'
    classes_to_retain=[26,49,51]
    comment='-reduced file size to 37- ugrad data exp1 '

if (db=='train_ugrad_noaug_500px'):
    image_list_file = r"C:\Z-Dataset\train_500px\png_snippets_q.txt"
    image_folder_in = r"C:\Z-Dataset\train_500px\\"
    image_folder_out = r"C:\Z-Dataset\tf_data\dataset_2\\"
    tfrecord_file_base = 'noaug_500px_train_i'
    tfrecord_len = 3000
    performed = '2025_11_10'
    data_set_type = 'Train Data'
    read_quality = True
    comment='ugrad data exp1 '


if (db=='test_ugrad_noaug_500px'):
    image_list_file = r"C:\Z-Dataset\test_500px\png_snippets_q.txt"
    image_folder_in = r"C:\Z-Dataset\test_500px\\"
    image_folder_out = r"C:\Z-Dataset\tf_data\dataset_2\\"
    tfrecord_file_base = 'noaug_500px_validation_i'
    tfrecord_len = 3000
    performed = '2025_11_10'
    data_set_type = 'Test Data'
    read_quality = True
    comment='ugrad data exp 1 '





#########################################################
# Reads the image list file
# image_folder = /path/to/images/
# format of the image_list_file  image.png[tab]label
# label should be an integer
# Returns a image list and the corresponding label list
#########################################################
def read_labeled_image_list(image_folder, image_list_file, read_qlty=False):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    if (read_qlty):
        qltys=[]
    for line in f:
        if (read_qlty):
            filename, label, qlty = line[:-1].split('\t')
            qltys.append(int(qlty))
        else:
            filename, label = line[:-1].split('\t')
        filenames.append(image_folder+filename)
        #print(label)
        labels.append(int(label))

    if (read_qlty):
        return filenames,labels, qltys
    else:
        return filenames, labels


#########################################################
# Reads the image list file
# image_folder = /path/to/images/
# format of the image_list_file  image.png[tab]index[tab]label[tab]xtf_file
# label should be an integer
# Returns a image list and the corresponding label list
#########################################################
def read_labeled_image_list_withIndex_and_xtf(image_folder, image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    indexes= []
    for line in f:
        filename,index, label,xtf_name = line[:-1].split(None,4)
        filenames.append(image_folder+filename)
        #print(label)
        labels.append(int(label))
        indexes.append(int(index))
    return filenames, labels,indexes


#########################################################
# Reads the image list file
# image_folder = /path/to/images/
# format of the image_list_file  image.png[tab]label
# label should be an integer
# Returns a image list and the corresponding label list
#########################################################
def read_labeled_image_list_withIndex(image_folder, image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    indexes= []
    for line in f:
        index, filename, label = line[:-1].split('\t')
        filenames.append(image_folder+filename)
        #print(label)
        labels.append(int(label))
        indexes.append(int(index))
    return filenames, labels,indexes


###################################################
#
####################################################
def filter_samples(im,lab,ql):
    im_out=[]
    lab_out=[]
    ql_out=[]

    if (class_filtering == 'retain'):
        for i, label in enumerate(lab):
            if (label in classes_to_retain):
                im_out.append(im[i])
                lab_out.append(label)
                ql_out.append(ql[i])

    if (class_filtering == 'filter'):
        for i, label in enumerate(lab):
            if not (label in classes_to_filter):
                im_out.append(im[i])
                lab_out.append(label)
                ql_out.append(ql[i])

    return im_out, lab_out, ql_out



########################################################
# Load image
########################################################
def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


########################################################
# Load image
########################################################
def image_size(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)

    height, width =img.shape[:2]
    return height, width


########################################################
# Convert data into features
########################################################

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



########################################################
# Write data into tfrecords
########################################################
if (read_quality):
    image_files, labels, qltys=read_labeled_image_list(image_folder_in, image_list_file, read_qlty=True)
    if (class_filtering is not None):
        image_files, labels, qltys=filter_samples(image_files,labels, qltys)
else:
    if (read_index):
        if (v_2024):
            image_files, labels, indexes = read_labeled_image_list_withIndex_and_xtf(image_folder_in, image_list_file)
        else:
            image_files, labels,indexes = read_labeled_image_list_withIndex(image_folder_in, image_list_file)
    else:
        image_files, labels=read_labeled_image_list(image_folder_in, image_list_file)


nimages=len(image_files)

j=0

while(1):

    tf_filename =image_folder_out + tfrecord_file_base + str(j)+ '.tfrecord'

    if not os.path.exists(image_folder_out):
        os.makedirs(image_folder_out)

    # open the TFRecords file
    #writer = tf.python_io.TFRecordWriter(tf_filename)
    writer = tf.io.TFRecordWriter(tf_filename)

    for i in range(tfrecord_len):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print (data_set_type, ': {}/{}'.format(i, tfrecord_len))
            sys.stdout.flush()

        # Load the image
        current_index=j*tfrecord_len+i
        if (current_index >= nimages):
            j=-1
            break

        #img_raw=img.tostring()
        img_raw=open(image_files[current_index],"rb").read()
        im_height, im_width = image_size(image_files[current_index])

        label = labels[current_index]



        if (read_index):
            file_index =indexes[current_index]
        else:
            file_index=current_index

        # Create a feature
        feature = {'image/class/label': _int64_feature(label-1),   # Matlab 1 based index to Python 0 based index
                   'image/index': _int64_feature(file_index),
                   'image/height': _int64_feature(im_height),
                   'image/width': _int64_feature(im_width),
                   'image/format': _bytes_feature(b'png'),
                   'image/encoded': _bytes_feature(tf.compat.as_bytes(img_raw))}
        if (read_quality):
            img_quality=qltys[current_index]
            feature['image/quality']= _int64_feature(img_quality)

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    if(j<0):
        break
    else:
        j=j+1

writer.close()
sys.stdout.flush()

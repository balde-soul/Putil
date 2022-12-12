# coding=utf-8
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', dest='RootDir', type=str, action='store', default='', help='specify the brainwash datasset root dir')
options = parser.parse_args()
import os
from PIL import Image
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

if options.RootDir == '':
    print('need root_dir specified')
    sys.exit(1)
train_idl = os.path.join(options.RootDir, 'brainwash_train.idl')
val_idl = os.path.join(options.RootDir, 'brainwash_val.idl')
test_idl = os.path.join(options.RootDir, 'brainwash_test.idl')
AnnoDir = os.path.join(options.RootDir, 'Annotations')
ImageDir = os.path.join(options.RootDir, 'JPEGImages')
if not os.path.exists(AnnoDir):
    os.mkdir(AnnoDir)
if not os.path.exists(ImageDir):
    os.mkdir(ImageDir)

def rename_image(target):
    return target.replace('/', '_')

def move_image(root_dir, to_dir):
    workdir, targetdir = os.path.split(root_dir)
    command = 'cd {2} && rdir={0} && ls $rdir | while read -r line || [ -n "$line" ]; do mv $rdir/$line {1}/{0}_$line; done'.format(targetdir, to_dir, workdir)
    print(command)
    os.system(command)

def save_xml(image_name,  boxes, save_dir, width=640, height=480, channel=3):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for box in boxes:
        (left,top,right,bottom) = (box[0],box[1],box[2],box[3])
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text="head"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    save_to = os.path.join(save_dir, image_name.replace('png', 'xml').replace('jpg', 'xml'))
    if os.path.exists(save_to):
        print('{0} is exist'.format(save_to))
        pass
    save_xml = os.path.join(save_to)
    with open(save_xml, 'wb') as f:
        f.write(xml)
    return

def extract_from_idl(idl_file):
    f1 = open(idl_file, 'r+')
    lines = f1.readlines()
    for i in range(len(lines)):

        line = lines[i]
        line = line.replace(":", ";")
        if line.split(";")[1] != "\n":
            #图片的目录和图片的名字
            img_path = rename_image(line.split(";")[0]).replace('"', '')

            #图片的坐标
            img_coor = line.split(";")[1]
            img_coor = img_coor.replace("(","")
            img_coor = img_coor.replace("),",";")
            img_coor = img_coor.replace(").", "")
            img_coor = img_coor.replace(")","")
            # print(img_coor)
            boxes=[]

            for coordinate in img_coor.split(";"):
                if coordinate != "\n" :
                    coor = [int(float(zb.strip())) for zb in coordinate.split(",")]
                    boxes.append(coor)

            save_xml(img_path, boxes, AnnoDir)

for i in ['brainwash_11_13_2014_images', 'brainwash_10_27_2014_images', 'brainwash_11_24_2014_images']:
    move_image(os.path.join(options.RootDir, i), os.path.join(options.RootDir, 'JPEGImages'))
    print('done')
    pass

if not os.path.exists(AnnoDir):
    os.mkdir(AnnoDir)
    pass
extract_from_idl(train_idl)
extract_from_idl(val_idl)
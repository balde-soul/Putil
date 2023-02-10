import xml.etree.ElementTree as ET
import os, copy, json, sys
import random, argparse, traceback
random.seed(1995)

options = argparse.ArgumentParser()
options.add_argument('--image_root', dest='ImageRoot', type=str, action='store', default='', help='指定图像存储根目录')
options.add_argument('--xml_root', dest='XmlRoot', type=str, default='', help='指定xml存储路径')
options.add_argument('--class_filter_and_onehot', dest='ClassFilterAndOnehot', type=str, nargs='+', help='指定类别过滤器，以及使用该过滤器进行独热码赋值，labelme标记并不指定对应类别的独热码')
options.add_argument('--class_reflect', dest='ClassReflect', type=str, default=[], nargs='+', help='目标类别重映射,用于多种标注类别混合成一种,格式为--class_reflect A,B C,D')
options.add_argument('--yolo_label_save_to', dest='YoloLabelSaveTo', type=str, default='', action='store', help='指定生成yolo格式的label的保存位置,如果不指定,则根据image_root来获取保存位置')
args = options.parse_args()
# <block_begin: debug
# @time 2023-01-11
# @author cjh
#args.ImageRoot = '/data/caojihua/data/20221208有带厨师服厨师帽/image_test/'
#args.XmlRoot = '/data/caojihua/data/20221208有带厨师服厨师帽/test/'
#args.ClassFilterAndOnehot = ['', 'head']
#args.YoloLabelSaveTo = '/data/caojihua/data/20221208有带厨师服厨师帽/label_test/'
# block_end: >
args.SetRate = [1.0]
args.SetName = ['all']
 
xml_root = os.path.abspath(args.XmlRoot)
image_root = os.path.abspath(args.ImageRoot)
split_save_to = os.path.abspath(os.path.split(os.path.abspath(image_root))[0])
if args.YoloLabelSaveTo == '':
    formated_save_to = os.path.join(os.path.split(os.path.abspath(image_root))[0], 'labels')
elif os.path.exists(os.path.split(os.path.dirname(args.YoloLabelSaveTo))[0]):
    formated_save_to = args.YoloLabelSaveTo
    if not os.path.exists(formated_save_to):
        os.mkdir(formated_save_to)
        pass
else:
    print('yolo_label_save_to: {0} is not not supported'.format(args.YoloLabelSaveTo))
    sys.exit(1)
set_rate = args.SetRate
set_name = args.SetName
class_filter_and_onehot = args.ClassFilterAndOnehot
class_reflect = {c: cs_index for cs_index, cs in enumerate([cr.split(',') for cr in args.ClassReflect]) for c in cs}
# 检查类型映射是否满足独热码
if class_reflect != {}:
    print('use class reflect')
    one_hot_extract = list(set([class_reflect[a] for a in class_filter_and_onehot]))
    if max(one_hot_extract) != (len(one_hot_extract) - 1):
        print('one_hot_extract: {0}'.format(one_hot_extract))
        print('class_filter_and_onehot: {0}'.format(class_filter_and_onehot))
        print('class_reflect: {0}'.format(class_reflect))
        raise RuntimeError('unfit')
        pass
    pass
else:
    print('use default class filter {0} for onehot'.format(class_filter_and_onehot))
    pass

if not os.path.exists(formated_save_to):  # 修改路径（最好使用全路径）
    os.makedirs(formated_save_to)  # 修改路径（最好使用全路径）

total_xml = os.listdir(xml_root)
all_index = list(range(len(total_xml)))
numbers = [int(len(all_index)* i) for i in args.SetRate]

# 分割数据集
set_index = list()
st = 0
ed = 0
random.shuffle(all_index)
for n in numbers:
    set_index.append(all_index[ed: ed + n])
    ed = ed + n

def convert(size, box):
    dw = 1./(size[0])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dw = 1./((size[0])+0.1)
    dh = 1./(size[1])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dh = 1./((size[0])+0.1)
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
##@brief
# @note
# @param[in]
# @param[in]
# @return 
def convert_annotation(xml_id, classes):
    in_file = open(os.path.join(xml_root, '{0}.xml'.format(xml_id)))
    out_file = open(os.path.join(formated_save_to, '{0}.txt'.format(xml_id)), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    image_name = root.find('filename').text
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    object_types = list()
    object_one_hot = dict()
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        object_type = obj.find('name').text
        object_types.append(object_type)
        # 非目标类别以及标记为困难标注类型的被忽略
        if (object_type not in classes or int(difficult)==1) and object_type not in class_reflect.keys():
            continue
        cls_id = classes.index(object_type) if object_type not in class_reflect.keys() else class_reflect[object_type]
        assert (cls_id == object_one_hot[object_type]) if object_type in object_one_hot.keys() else True, 'cls_id switch check failed!'
        object_one_hot[object_type] = cls_id
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    return object_types, object_one_hot, image_name
 
object_types_set = list()
object_one_hot_set = dict()
ignored = list()
for index, sn in zip(set_index, set_name):
    xml_ids = [total_xml[i].replace('.xml', '') for i in index]
    for xml_id in xml_ids:
        try:
            object_types, object_one_hot, image_name = convert_annotation(xml_id, class_filter_and_onehot)
            object_types_set += object_types
            for ot in object_types:
                if ot in object_one_hot_set.keys():
                    assert object_one_hot_set[ot] == object_one_hot[ot]
                    pass
                elif ot in object_one_hot.keys():
                    object_one_hot_set[ot] = object_one_hot[ot]
                    pass
                else:
                    ignored.append(ot)
                pass
            pass
        except Exception as ex:
            print(ex.args)
            raise ex
            continue
        pass
    pass
print(', '.join(list(set(object_types_set))))

with open(os.path.join(split_save_to, 'info.json'), 'w') as fp:
    fp.write(json.dumps({
        'class_filter_and_onehot': class_filter_and_onehot,
        'class_reflect': class_reflect,
        'final_reflect': object_one_hot_set,
        'has_class': list(set(object_types_set)),
        'ignored': list(set(ignored))
    }, indent='\t'))
    pass

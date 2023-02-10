# coding=utf-8
import argparse, os, fnmatch, shutil, os, glob, cv2, copy, sys
import numpy as np
from scipy.io import loadmat
from collections import defaultdict
from lxml import etree, objectify
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from Putil.tools.data_process.voc_util import VOCToolSet

def vbb_anno2dict(vbb_file, cam_id):
    # 通过os.path.basename获得路径的最后部分“文件名.扩展名”
    # 通过os.path.splitext获得文件名
    filename = os.path.splitext(os.path.basename(vbb_file))[0]

    # 定义字典对象annos
    annos = defaultdict(dict)
    vbb = loadmat(vbb_file)
    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]  # 可查看所有类别
    # person index
    person_index_list = np.where(np.array(objLbl) == "person")[0]  # 只选取类别为‘person’的xml
    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            frame_name = str(cam_id) + "_" + str(filename) + "_" + str(frame_id + 1) + ".jpg"
            annos[frame_name] = defaultdict(list)
            annos[frame_name]["id"] = frame_name
            annos[frame_name]["label"] = "person"
            for id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                id = int(id[0][0]) - 1  # for matlab start from 1 not 0
                if not id in person_index_list:  # only use bbox whose label is person
                    continue
                pos = pos[0].tolist()
                occl = int(occl[0][0])
                annos[frame_name]["occlusion"].append(occl)
                annos[frame_name]["bbox"].append(pos)
            if not annos[frame_name]["bbox"]:
                del annos[frame_name]
    print(annos)
    return annos


def instance2xml_base(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/person'),
        E.filename(anno['id']),
        E.source(
            E.database('Caltech pedestrian'),
            E.annotation('Caltech pedestrian'),
            E.image('Caltech pedestrian'),
            E.url('None')
        ),
        E.size(
            E.width(640),
            E.height(480),
            E.depth(3)
        ),
        E.segmented(0),
    )
    for index, bbox in enumerate(anno['bbox']):
        bbox = [float(x) for x in bbox]
        if bbox_type == 'xyxy':
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
        else:
            xmin, ymin, xmax, ymax = bbox
        E = objectify.ElementMaker(annotate=False)
        anno_tree.append(
            E.object(
                E.name(anno['label']),
                E.bndbox(
                    E.xmin(xmin),
                    E.ymin(ymin),
                    E.xmax(xmax),
                    E.ymax(ymax)
                ),
                E.difficult(0),
                E.occlusion(anno["occlusion"][index])
            )
        )
    return anno_tree


def parse_anno_file(vbb_inputdir, vbb_outputdir):
    # annotation sub-directories in hda annotation input directory
    assert os.path.exists(vbb_inputdir)
    sub_dirs = os.listdir(vbb_inputdir)  # 对应set00,set01...
    for sub_dir in sub_dirs:
        print("Parsing annotations of camera: ", sub_dir)
        cam_id = sub_dir  # set00 set01等
        # 获取某一个子set下面的所有vbb文件
        vbb_files = glob.glob(os.path.join(vbb_inputdir, sub_dir, "*.vbb"))
        for vbb_file in vbb_files:
            # 返回一个vbb文件中所有的帧的标注结果
            annos = vbb_anno2dict(vbb_file, cam_id)

            if annos:
                # 组成xml文件的存储文件夹，形如“/Users/chenguanghao/Desktop/Caltech/xmlresult/”
                vbb_outdir = vbb_outputdir

                # 如果不存在
                if not os.path.exists(vbb_outdir):
                    os.makedirs(vbb_outdir)

                for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                    if "bbox" in anno:
                        anno_tree = instance2xml_base(anno)
                        outfile = os.path.join(vbb_outdir, os.path.splitext(filename)[0] + ".xml")
                        print("Generating annotation xml file of picture: ", filename)
                        # 生成最终的xml文件，对应一张图片
                        etree.ElementTree(anno_tree).write(outfile, pretty_print=True)

class Caltech:
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._annotation_dir = os.path.join(self._root_dir, 'annotations')
        self._splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"
        self._set_names = ['set01', 'set02', 'set03', 'set04', 'set05', 'set06', 'set07', 'set08', 'set09', 'set10']
        pass

    def get_set_names(self):
        return copy.deepcopy(self._set_names)

    def get_ids(self, set_name):
        files = os.listdir(os.path.join(self._root_dir, set_name))
        return [f.replace('.seq', '') for f in files]

    def get_images(self, set_name, _id):
        f = open(os.path.join(self._root_dir, os.path.join(set_name, '{0}.seq'.format(_id))), 'rb+')
        string = f.read().decode('latin-1')
        # split .seq file into segment with the image prefix
        strlist = string.split(self._splitstring)
        f.close()
        return strlist[1:]

    def get_image_frame(self, images, frame):
        return images[frame]

    def image_save_to_jpg(self, img, jpg_file):
        i = open(jpg_file, 'wb+')
        i.write(self._splitstring.encode('latin-1'))
        i.write(img.encode('latin-1'))
        i.close()
        pass

    def get_labels(self, set_name, _id):
        annos = defaultdict(dict)
        vbb = loadmat(os.path.join(self._annotation_dir, os.path.join(set_name, '{0}.vbb'.format(_id))))
        # object info in each frame: id, pos, occlusion, lock, posv
        objLists = vbb['A'][0][0][1][0]
        objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
        return vbb, objLists, objLbl

    def get_label(self, labels, frame):
        return labels[frame]

def to_voc(options):
    if not os.path.exists(options.SaveTo):
        os.mkdir(options.SaveTo)
        pass
    JPEGImagesDir = os.path.join(options.SaveTo, 'JPEGImages')
    if not os.path.exists(JPEGImagesDir):
        os.mkdir(JPEGImagesDir)
    AnnotationsDir = os.path.join(options.SaveTo, 'Annotations')
    if not os.path.exists(AnnotationsDir):
        os.mkdir(AnnotationsDir)
    image_save_to = JPEGImagesDir
    annotation_save_to = AnnotationsDir

    caltech = Caltech(options.CaltechRoot)

    set_names = caltech.get_set_names()
    ids = {sn: caltech.get_ids(sn) for sn in set_names}
    # read .seq file and save the images into the savepath

    for sn, ids in ids.items():
        for _id in ids:
            images = caltech.get_images(sn, _id)

            vbb, objLists, objLbl = caltech.get_labels(sn, _id)
            person_index_list = np.where(np.array(objLbl) == "person")[0]  # 只选取类别为‘person’的xml

            for frame, (_, obj) in enumerate(zip(images, objLists)):
                if int(frame * options.SampleRate) < int((frame + 1) * options.SampleRate):
                    frame_id = '{0}-{1}-{2}'.format(sn, _id, frame)
                    object_cells = list()
                    if len(obj) > 0:
                        for id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                            id = int(id[0][0]) - 1  # for matlab start from 1 not 0
                            if not id in person_index_list:  # only use bbox whose label is person
                                continue
                            pos = pos[0].tolist()
                            occl = int(occl[0][0])
                            object_cells.append({
                                'name': 'person',
                                'bbox': {
                                    'xmin': pos[0], 
                                    'ymin': pos[1], 
                                    'xmax': pos[0] + pos[2], 
                                    'ymax': pos[1] + pos[3]
                                }
                            })
                            pass
                        pass
                    if len(object_cells) != 0:
                        image = caltech.get_image_frame(images, frame)
                        imgdir = os.path.join(image_save_to, '{0}.jpg'.format(frame_id))
                        caltech.image_save_to_jpg(image, imgdir)
                        xmldir = os.path.join(annotation_save_to, '{0}.xml'.format(frame_id))
                        VOCToolSet.generate_empty_xml(imgdir, xmldir)
                        VOCToolSet.append_object(xmldir, object_cells)
                    pass
            pass
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_to', dest='SaveTo', type=str, default='', action='store', help='指定提取保存位置')
    parser.add_argument('--caltech_root', dest='CaltechRoot', type=str, default='', action='store', help='指定Caltech数据集的根目录')
    parser.add_argument('--exclude_no_person', dest='ExcludeNoPerson', action='store_true', help='当指定时，去除没有人员的样本帧')
    parser.add_argument('--sample_rate', dest='SampleRate', type=float, default=1.0, action='store', help='指定帧采样率')
    options = parser.parse_args()

    # <block_begin: 测试
    # @time 2023-02-02
    # @author cjh
    #options.SaveTo = '/data/caojihua/data/Caltech/VOC/'
    #options.CaltechRoot = '/data/caojihua/data/Caltech/Original/'
    #options.SampleRate = 0.033
    # block_end: >
    to_voc(options)
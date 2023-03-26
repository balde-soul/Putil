# coding=utf-8
import re, os, cv2
from xml.etree.ElementTree import ElementTree, Element
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

##@brief
# @note
#   * object单元数据:
#     * [
#           {
#               'name': str, 
#               'bbox': {'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int}, 
#               'pose': str, 
#               'truncated': int, 
#               'difficult': int, 
#               'prob': '目标置信度'
#           }
#       ]
#   * fid:图像去除后缀的名称
#   * image_info:xml标记文件去除object信息之后的图像信息
# @time 2023-01-04
# @author cjh
class VOCToolSet:
    def __init__(self):
        pass

    @staticmethod
    def image2id(imgdir):
        if not os.path.isfile(imgdir):
            raise RuntimeError('imgdir: {0} is not file'.format(imgdir))
            pass

        imgdir = os.path.split(imgdir)[-1]
        return imgdir[::-1][re.search('\.', imgdir[::-1]).end():][::-1]

    @staticmethod
    def id2xml(fid):
        return '{0}.xml'.format(fid)

    @staticmethod
    def xml2id(xmldir):
        if not os.path.isfile(xmldir):
            raise RuntimeError('xmldir: {0} is not file'.format(xmldir))
        xmldir = os.path.split(xmldir)[-1]
        return xmldir.replace('.xml', '')

    ##@brief
    # @note 从xml文件中提取出object单元数据
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-01-04
    # @author cjh
    @staticmethod
    def extract_object(xmldir):
        object_cells = list()
        in_file = open(xmldir)
        tree = ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            object_cell = dict()
            object_cell['difficult'] = obj.find('difficult').text
            object_cell['name'] = obj.find('name').text
            xmlbox = obj.find('bndbox')
            object_cell['bbox'] = {'xmin': int(xmlbox.find('xmin').text), 'xmax': int(xmlbox.find('xmax').text), 'ymin': int(xmlbox.find('ymin').text), 'ymax': int(xmlbox.find('ymax').text)}
            object_cells.append(object_cell)
        return object_cells

    ##@brief
    # @note 多个目标单元数据打包成VOC xml子节点
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-01-04
    # @author cjh
    @staticmethod
    def append_object(xmldir, object_cells):
        with open(xmldir, 'r', encoding='utf-8') as fp:
            dom = minidom.parse(fp)
        node_root = ET.fromstring(dom.toxml())
        #in_file = open(xmldir)
        #tree = ET.parse(in_file)
        #node_root = tree.getroot()
        for oc in object_cells:
            node_object = Element('object')
            node_name = Element('name')
            node_name.text = str(oc['name'])
            node_object.append(node_name)
            node_prob = Element('prob')
            node_prob.text = oc.get('prob', '')
            node_object.append(node_prob)
            node_pose = Element('pose')
            node_pose.text = oc.get('pose', "Unspecified")
            node_object.append(node_pose)
            node_truncated = Element('truncated')
            node_truncated.text = oc.get('truncated', '0')
            node_object.append(node_truncated)
            node_difficult = Element('difficult')
            node_difficult.text = oc.get('difficult', '0')
            node_object.append(node_difficult)
            node_bndbox = Element('bndbox')
            node_xmin = Element('xmin')
            node_xmin.text = str(int(oc['bbox']['xmin'] + 0.5))
            node_bndbox.append(node_xmin)
            node_ymin = Element('ymin')
            node_ymin.text = str(int(oc['bbox']['ymin'] + 0.5))
            node_bndbox.append(node_ymin)
            node_xmax = Element('xmax')
            node_xmax.text = str(int(oc['bbox']['xmax'] + 0.5))
            node_bndbox.append(node_xmax)
            node_ymax = Element('ymax')
            node_ymax.text = str(int(oc['bbox']['ymax'] + 0.5))
            node_bndbox.append(node_ymax)
            node_object.append(node_bndbox)
            node_root.append(node_object)
        xml = ET.tostring(node_root)
        dom = minidom.parseString(xml)
        with open(xmldir, 'w', encoding='utf-8') as f:
            dom.writexml(f, '', '\t', '\n', 'utf-8')
        #xml = tostring(node_root, pretty_print=True, encoding='utf-8')  # 格式化显示，该换行的换行
        #file_object = open(xmldir, 'wb')
        #file_object.write(xml)
        #file_object.close()
        pass

    ##@brief
    # @note 生成空的xml标记文件
    # @param[in] imgdir:可以是图像路径，也可以是image_info
    # @param[in]
    # @return 
    # @time 2023-01-04
    # @author cjh
    @staticmethod
    def generate_empty_xml(imgdir, xmldir):
        if isinstance(imgdir, str):
            img = cv2.imdecode(np.fromfile(imgdir, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            imh, imw = img.shape[0: 2]
            depth = '1' if len(img.shape) == 2 else '{0}'.format(img.shape[-1])
            filename = os.path.split(imgdir)[-1]
            path = imgdir
        elif isinstance(imgdir, dict):
            imh, imw, depth, filename, path = imgdir['height'], imgdir['width'], imgdir['depth'], imgdir['filename'], imgdir['path']
            pass
        tree = ElementTree()
        node_root = Element('annotation')
        node_folder = Element('folder')
        node_folder.text = 'image'
        node_root.append(node_folder)
        node_filename = Element('filename')
        node_filename.text = filename
        node_root.append(node_filename)
        node_filepath = Element('path')
        node_filepath.text = path
        node_root.append(node_filepath)
        node_size = Element('size')
        node_width = Element('width')
        node_width.text = str(imw)
        node_size.append(node_width)
        node_height = Element('height')
        node_height.text = str(imh)
        node_size.append(node_height)
        node_depth = Element('depth')
        node_depth.text = str(depth)
        node_size.append(node_depth)
        node_root.append(node_size)
        tree._setroot(node_root)
        xml = ET.tostring(node_root)
        dom = minidom.parseString(xml)
        with open(xmldir, 'w', encoding='utf-8') as f:
            dom.writexml(f, '', '\t', '\n', 'utf-8')
        #node_folder = SubElement(node_root, 'folder')
        #node_folder.text = 'imge'
        #xml = tostring(node_root, pretty_print=True, encoding='utf-8')  # 格式化显示，该换行的换行
        #file_object = open(xmldir, 'wb')
        #file_object.write(xml)
        #file_object.close()
        pass

    ##@brief 从标记文件中获取image信息
    # @note
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-01-04
    # @author cjh
    @staticmethod
    def get_image_info(xmldir):
        ret = dict()
        in_file = open(xmldir)
        tree = ET.parse(in_file)
        root = tree.getroot()
        path = root.find('path') # path有无无所谓
        ret['path'] = path.text if path is not None else ''
        ret['filename'] = root.find('filename').text
        size = root.find('size')
        ret['width'] = int(size.find('width').text)
        ret['height'] = int(size.find('height').text)
        ret['depth'] = int(size.find('depth').text)
        return ret

    @staticmethod
    def extract_image_info(xmldir):
        return VOCToolSet.get_image_info(xmldir)
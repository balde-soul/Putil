# coding=utf-8
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
parser = argparse.ArgumentParser()
parser.add_argument('--image_root', dest='ImageRoot', type=str, default='', action='store', help='指定图像根目录')
parser.add_argument('--xml_root', dest='XmlRoot', type=str, default='', action='store', help='指定xml根目录')
parser.add_argument('--target_class', dest='TargetClass', type=str, nargs='+', default=[], help='目标类别')
parser.add_argument('--sep_contain_target_and_not', dest='SepContainTargetAndNot', action='store_true', default=False, help='指定target_class,将数据集分为包含target_class中某一类和不包含target_class中的任一类两个集合，\n并将图像集合集合保存到\'-\'.join(target_class)-contain.txt和\'-\'.join(target_class)-no_contain.txt文件中')
parser.add_argument('--save_to', dest='SaveTo', type=str, action='store', default='', help='voc_dataset_clean 输出文件的保存路径,为')
parser.add_argument('--attachabled', dest='Attachabled', action='store_true', help='是否链接debug')
options = parser.parse_args()
if options.Attachabled:
    import ptvsd
    host = '127.0.0.1'
    port = 12345
    ptvsd.enable_attach(address=(host, port), redirect_output=True)
    ptvsd.wait_for_attach()
    pass

from Putil.path import touch_dir
from Putil.tools.data_process.voc_util import VOCToolSet

if not os.path.exists(options.SaveTo):
    touch_dir(options.SaveTo)
    pass

if options.XmlRoot == '':
    print('xml_root should be specify or the path({0}) of xml_root spcified is not exist'.format(options.XmlRoot))
    sys.exit(1)
    pass

if options.ImageRoot == '' or not os.path.exists(options.ImageRoot):
    print('image_root should be specify or the path({0}) of image_root spcified is not exist'.format(options.ImageRoot))
    sys.exit(1)
    pass

if options.SepContainTargetAndNot:
    contain = list()
    not_contain = list()
    print('running sep_contain_target_and_not:')
    imgs = os.listdir(options.ImageRoot)
    print('total set: {0}'.format(len(imgs)))
    for i in imgs:
        has = False
        fid = VOCToolSet.image2id(os.path.join(options.ImageRoot, i))
        xmldir = os.path.join(options.XmlRoot, VOCToolSet.id2xml(fid))
        if not os.path.exists(xmldir):
            not_contain.append(i)
            continue 

        objs = VOCToolSet.extract_object(xmldir)
        for obj in objs:
            if obj['name'] in options.TargetClass:
                contain.append(i)
                has = True
                break 
        if not has:
            not_contain.append(i)
        pass
    print('contain {0} set: {1}'.format('-'.join(options.TargetClass), len(contain)))
    print('not contain {0} set: {1}'.format('-'.join(options.TargetClass), len(not_contain)))
    with open(os.path.join(options.SaveTo, '{0}-contain.txt'.format('-'.join(options.TargetClass))), 'w') as fp:
        fp.writelines('\n'.join(contain))
        fp.writelines('\n')
        pass
    with open(os.path.join(options.SaveTo, 'contain.txt'), 'w') as fp:
        fp.writelines('\n'.join(contain))
        fp.writelines('\n')
        pass
    with open(os.path.join(options.SaveTo, '{0}-not-contain.txt'.format('-'.join(options.TargetClass))), 'w') as fp:
        fp.writelines('\n'.join(not_contain))
        fp.writelines('\n')
        pass
    with open(os.path.join(options.SaveTo, 'not-contain.txt'), 'w') as fp:
        fp.writelines('\n'.join(not_contain))
        fp.writelines('\n')
        pass
    print('done')
    pass
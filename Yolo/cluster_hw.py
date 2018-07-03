# coding = utf-8
import json


class yolo_cluster:
    """
    yolo矩形框先验高宽计算
    多种类型聚类-->生成item-mIoU曲线
        -->生成多种聚类结果json文件
        type_name{
            item_0:{
                h_w_0:
            }
            item_1:{
                h_w_0: [H, W]
                h_w_1: [H, W]
            }
        }
        type_name{...}
    """
    def __init__(self, array):
        self._GT = array
        pass

    def __kmeans(self, **options):
        pass

    def __cluster(self, type, save_path, **options):
        if type=='kmeans':
            config, item, mIoU = self.__kmeans()
            json_name = save_path + '-kmeans.json'
            j = json.dumps(config)
            with open(json_name, 'w') as fp:
                fp.write(j)
            pass
        pass

    def  analysis(self, *type, **options):
        pass

    pass


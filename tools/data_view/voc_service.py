# coding-utf8
import sys, os, argparse, fastapi, json, time. uvicorn, threading, copy, traceback, datetime
from pydantic import BaseModel
import logging as logger

parser = argparser.ArgumentParser()
## action
### 增加数据集
parser.add_argument('--add_dataset', dest='AddDataset', action='store_true', help='指定动作，往服务中增加数据集，一个服务可以支撑多个数据集的读取')
parser.add_argument('--add_dataset_tag', dest='AddDatasetTag', action='store', type=str, default='', help='指定增加数据集的标记名称')
parser.add_argument('--image_root', dest='ImageRoot', type=str, action='store', default='', help='指定图像存储目录')
parser.add_argument('--xml_root', dest='XmlRoot', type=str, action='store', default='', help='指定xml存储路径')
### 启动服务
parser.add_argument('--start_up', dest='StartUp', action='store_true', help='起服务动作') 
parser.add_argument('--config_file', dest='ConfigFile', type=str, action='store', default='', help='指定配置文件，默认为空')
parser.add_argument('--port', dest='Port', type=int, action='store', default='8080', help='指定服务的port')
parser.add_argument('--ip', dest='IP', type=str, action='store', default='0.0.0.0', help='指定服务的ip')
### 把当前配置保存倒配置文件
options = parser.parse_args()

def add_dataset():
    pass

if options.AddDataset:
    pass

if options.StartUp:
    app = fastapi.FastAPI()

    class Manager:
        def __init__(self):
            self._datasets = dict()
            self._datasets_lock = threading.Lock()
            pass

        def get_image_objects(self, dataset, imgid):
            pass

        def get_datasets(self):
            self._datasets_lock.required()
            try:
                return copy.deepcopy(self._datasets)
            except Exception as ex:
                raise ex
            finally:
                self._datasets_lock.release()
            pass

        def add_dataset(self, tag, image_root, xml_root):
            self._datasets_lock.required()
            try:
                if tag in self._datasets.keys():
                    tag = '{0}-{1}'.format(tag,datetime.datetime.strptime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S-%f'))
                    pass
                self._datasets[tag] = {
                    'image_root': image_root,
                    'xml_root': xml_root,
                    'info': ''
                }
            except Exception as ex:
                logger.warning()
                raise ex
            finally:
                self._datasets_lock.release()
            pass
        pass

    manager = Manager()
    
    # todo: 通过配置文件加入datasets
    if options.ConfigFile != '':
        pass
    
    @app.post('/get_datasets')
    async def get_datasets():
        ret = {
            'Status': True,
            'msg': '',
            'data': dict()
        }
        try:
            data = manager.get_datasets()
            ret['Status'] = True
            ret['msg'] = 'ok'
            ret['data'] = data
        except Exception as ex:
            ret['Status'] = False
            ret['msg'] = 'get_datasets: {0}\ntrackback: {1}'.format(ex.args, traceback.format_exc())
            logger.warning('get_datasets: {0}\ntrackback: {1}'.format(ex.args, traceback.format_exc()))
            pass
        finally:
            return json.dumps(ret)
            pass
        pass

    class ShowLabelRe(BaseModel):
        taskid: str
        dataset: str
        imageid: str

    @app.post('/show_label')
    async def show_label(request: MyRequest):
        ret = {
            'taskid': '',
            'Status': True,
            'msg': '',
            'data': dict()
        }
        ret['taskid'] = request.taskid
        try:
            data = manager.get_image_objects(dataset, imgid)
            ret['Status'] = True
            ret['msg'] = 'ok'
            ret['data'] = data
        except Exception as ex:
            logger.warning('get_datasets: {0}\ntrackback: {1}'.format(ex.args, traceback.format_exc()))
            ret['Status'] = False
            ret['msg'] = 'get_datasets: {0}\ntrackback: {1}'.format(ex.args, traceback.format_exc())
        finally:
            return json.dumps(ret)
        pass
    uvicorn.run(app='api:app', host=options.IP, port=options.Port)
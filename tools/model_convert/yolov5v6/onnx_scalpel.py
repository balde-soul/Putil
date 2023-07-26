##@FileName onnx_scalpel.py
# @Note 用于修改yolov5v6导出的onnx模型
# @Author cjh
# @Time 2023-05-06
# coding=utf-8

##@brief 使用Matcher匹配子图
# @note 记录入口与出口
# @time 2023-05-07
# @author cjh
class Matcher:
    def __init__(self):
        pass
    pass

##@brief 使用Replacer替换子图
# @note 提供子图
# @time 2023-05-07
# @author cjh
class Replacer:
    def __init__(self):
        pass
    pass

class Scalpel:
    def __init__(self):
        pass

    def operate(self, onnx_in, onnx_out):
        pass
    pass

#In[]
from platform import node
import onnx
with open('/data/caojihua/Project/ChuShiMaoJianCe/best.pt.MV6E2E353.onnx', 'rb') as fp:
    og = onnx.load(fp)
#In[]
finall_op = [n for n in og.graph.node if n.output[0]=='output']
assert len(finall_op) == 1
fo = finall_op[0]
output_dict = {o: i for i, n in enumerate(og.graph.node) for o in n.output}
for fo in fo.input:
    pass
#In[]
import onnx
import onnx_graphsurgeon as ogs

# <block_begin: 
# @time 2023-05-08
# @author cjh
'''
# ints:[class_count, input_w, input_h, max_output_object_count, is_segmentation]
yolon_netinfo = onnx.AttributeProto(name='netinfo', ints=[5, 640, 640, 100, 0])
test_weight = onnx.TensorProto(dims=[2], data_type=1, float_data=[1.0, 2.0], name='test_bias', data_location=0)
#yolon_kernels = onnx.AttributeProto(name='kernels', )
ogs.
yolon = onnx.NodeProto(
    op_type='YoloLayer_TRT',
    domain='test',
    name='yolo_1',
    input=['fe1', 'fe2', 'fe3', 'test_bias'],
    output=['out'],
    attribute=[yolon_netinfo]
    )
'''
# block_end: >

ginputs = list()
fe_tensor_type = onnx.TypeProto.Tensor(
    elem_type=1, 
    shape=onnx.TensorShapeProto(dim=[onnx.TensorShapeProto.Dimension(dim_value=1), 
onnx.TensorShapeProto.Dimension(dim_value=150), 
onnx.TensorShapeProto.Dimension(dim_value=32),
onnx.TensorShapeProto.Dimension(dim_value=32)]))
fe_type = onnx.TypeProto(tensor_type=fe_tensor_type, denotation='feature_1')
fe = onnx.ValueInfoProto(type=fe_type, name='fe1', doc_string='l1 feature map')
ginputs.append(fe)
fe_shape = onnx.TensorShapeProto(dim=[onnx.TensorShapeProto.Dimension(dim_value=1), 
onnx.TensorShapeProto.Dimension(dim_value=150), 
onnx.TensorShapeProto.Dimension(dim_value=32),
onnx.TensorShapeProto.Dimension(dim_value=32)])
fe_tensor_type = onnx.TypeProto.Tensor(elem_type=1, shape=fe_shape)
fe_type = onnx.TypeProto(tensor_type=fe_tensor_type, denotation='feature_2')
fe = onnx.ValueInfoProto(type=fe_type, name='fe2', doc_string='l2 feature map')
ginputs.append(fe)
fe_shape = onnx.TensorShapeProto(dim=[onnx.TensorShapeProto.Dimension(dim_value=1), 
onnx.TensorShapeProto.Dimension(dim_value=150), 
onnx.TensorShapeProto.Dimension(dim_value=32),
onnx.TensorShapeProto.Dimension(dim_value=32)])
fe_tensor_type = onnx.TypeProto.Tensor(elem_type=1, shape=fe_shape)
fe_type = onnx.TypeProto(tensor_type=fe_tensor_type, denotation='feature_3')
fe = onnx.ValueInfoProto(type=fe_type, name='fe3', doc_string='l3 feature map')
ginputs.append(fe)
ou_shape = onnx.TensorShapeProto(dim=[onnx.TensorShapeProto.Dimension(dim_value=1), 
onnx.TensorShapeProto.Dimension(dim_value=150), 
onnx.TensorShapeProto.Dimension(dim_value=32),
onnx.TensorShapeProto.Dimension(dim_value=32)])
ou_tensor_type = onnx.TypeProto.Tensor(elem_type=1, shape=ou_shape)
ou_type = onnx.TypeProto(tensor_type=ou_tensor_type, denotation='finall_output')
ou = onnx.ValueInfoProto(type=ou_type, name='out', doc_string='the output')

yolon_netinfo = onnx.AttributeProto(name='netinfo', type=onnx.AttributeProto.AttributeType.INTS, ints=[5, 640, 640, 100, 0])
yolon_test_bias = onnx.AttributeProto(name='test_bias', type=onnx.AttributeProto.AttributeType.FLOATS, floats=[1.0, 0.1])
test_weight = onnx.TensorProto(dims=[2], data_type=1, float_data=[1.0, 2.0], name='test_bias', data_location=0)
yolon = onnx.NodeProto(
    op_type='YoloLayer_TRT',
    domain='test',
    name='yolo_1',
    input=['fe1', 'fe2', 'fe3', 'test_bias'],
    output=['out'],
    attribute=[yolon_netinfo, yolon_test_bias]
    )
#yolon = ogs.Node(
#    op='YoloLayer_TRT', 
#    name='yolo_1', 
#    attrs={'netinfo': [5, 640, 640, 100, 0]}, 
#    inputs=['fe1', 'fe2', 'fe3'] + ['test_bias'],
#    outputs=['out'],
#)

graph = onnx.GraphProto(
    node=[yolon],
    output=[ou],
    initializer=[test_weight],
    input=ginputs
)

model = onnx.ModelProto(
    model_version=0,
    ir_version=6,
    opset_import=[
        onnx.OperatorSetIdProto(
            domain='',
            version=13)
            ],
    producer_name='pytorch_scalpel',
    domain='',
    graph=graph
)

onnx_file = './test_yolo.onnx'
with open(onnx_file, 'wb') as fp:
    fp.write(model.SerializeToString())
with open(onnx_file, 'rb') as fp:
    rm = onnx.ModelProto()
    rm.ParseFromString(fp.read())
    print(rm)

#In[]
from ctypes import *
import os
yoloplu = cdll.LoadLibrary('/data/caojihua/Project/ProjectBase/tensorrt/yolov5/build/libmyplugins.so')

import tensorrt as trt
TRT_LOGGER = trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open(onnx_file, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            else:
                config = builder.create_builder_config()
                #config.max_workspace_size =( 1 << 20 ) * 3 * 1024 # 3GB，可以根据需求改的更大
                #builder.max_batch_size = 1
                builder.fp32_mode = True
                # generate TensorRT engine optimized for the target platform
                print('Building an engine...')
                engine = builder.build_cuda_engine(network)
                context = engine.create_execution_context()
                print("Completed creating Engine")
    pass
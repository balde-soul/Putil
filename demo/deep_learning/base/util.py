import os 
from enum import Enum

def generate_model_element(sign):
    '''
     @brief use the element from the result of get_all_model to generate the target model name
     @ret dict represent the information useful in evaluate and test
    '''
    return {'checkpoint': generate_checkpoint_name(sign), 'deploy': generate_deploy_name(sign)}

def generate_model_sign(file_name):
    return file_name.split('.')[0].split('-')[0]

def target_model_filter(file_name):
    '''
     @brief check the file_name is the file or not
     @ret bool
    '''
    if file_name.split('.')[-1] == 'pt':
        return True
    else:
        return False

def get_all_model(target_path):
    '''
     @brief 
    '''
    elements = list()
    signs = list()
    files = os.listdir()
    for _file in files:
        if target_model_filter(_file) is True:
            signs.append(int(generate_model_sign(_file)))
        else:
            continue
    signs = sorted(temp_signs)
    for sign in signs[::-1]:
        me = generate_model_element(sign)
        elements.append(me)
    return {'sign': signs, 'element': elements}

def generate_deploy_name(epoch):
    return '{}-traced_model-jit.pt'.format(epoch)

def generate_checkpoint_name(epoch):
    return '{}.pkl'.format(epoch)

class Stage(Enum):
    Train=0
    TrainEvaluate=1
    Evaluate=2
    Test=3
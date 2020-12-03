import os 
from enum import Enum
import torch
from torch.nn import Module


class TemplateModelDecodeCombine(Module):
    def __init__(self, model, decode):
        self._model = model
        self._decode = decode
        pass
    pass


def torch_generate_model_element(epoch):
    '''
     @brief use the element from the result of get_all_model to generate the target model name
     @ret dict represent the information useful in evaluate and test
    '''
    return {'checkpoint': generate_checkpoint_name(epoch), 'deploy': generate_deploy_name(epoch), \
        'save': generate_save_name(epoch)}

def torch_generate_model_epoch(file_name):
    return file_name.split('.')[0].split('-')[0]


def torch_target_model_filter(file_name):
    '''
     @brief check the file_name is the file or not
     @ret bool
    '''
    if file_name.split('.')[-1] == 'pt':
        return True
    else:
        return False


def torch_get_all_model(target_path):
    '''
     @brief 
    '''
    elements = list()
    epochs = list()
    files = os.listdir()
    for _file in files:
        if target_model_filter(_file) is True:
            epochs.append(int(generate_model_epoch(_file)))
        else:
            continue
    epochs = sorted(temp_epochs)
    for epoch in epochs[::-1]:
        me = generate_model_element(epoch)
        elements.append(me)
    return {'epochs': epochs, 'elements': elements}


def torch_generate_deploy_name(epoch):
    return '{}-traced_model-jit.pt'.format(epoch)


def torch_generate_checkpoint_name(epoch):
    return '{}.pkl'.format(epoch)


def torch_generate_save_name(epoch):
    return '{}.pth'.format(epoch)


def torch_deploy(model, decode, template_model_decode_combine, input_example, epoch, full_path):
    # :use JIT to deploy a model
    '''
     @brief deploy model using in other language, such as c++, lua
     @note use JIT to deploy the model, we should combine the model and decode in to one Module
     @param[in] model the model
     @param[in] decode the decode
     @param[in] template_model_decode_combine the Module which inherit from TemplateModelDecodeCombine
     @param[in] input_example the input_example
     @param[in] epoch
     @param[in] full_path
    '''
    traced_script_module = torch.jit.trace(template_model_decode_combine(model, decode), input_example)
    traced_script_module.save(os.path.join(full_path, generate_deploy_name(epoch)))
    pass


def torch_save(model, decode, template_model_decode_combine, epoch, full_path):
    '''
     @brief deploy model using in evaluate stage
     @note use torch.save to save the model, we should combine the model and decode in to one Module
     @param[in] model the model
     @param[in] decode the decode
     @param[in] template_model_decode_combine the Module which inherit from TemplateModelDecodeCombine
     @param[in] epoch
     @param[in] full_path
    '''
    torch.save(template_model_decode_combine(model, decode), os.path.join(full_path, generate_save_name(epoch)))
    pass


def torch_checkpoint(epoch, full_path, *kargs, **kwargs):
    '''
     @brief deploy model using in continue training
     @note use torch.save to save the state_dict
     @param[in] model the model
     @param[in] decode the decode
     @param[in] template_model_decode_combine the Module which inherit from TemplateModelDecodeCombine
     @param[in] epoch
     @param[in] full_path
    '''
    state_dict = {key: value.state_dict() for key, value in kwargs}
    torch.save(state_dict, os.path.join(full_path, generate_checkpoint_name(epoch)))
    pass


def torch_load_saved(epoch, full_path):
    model = torch.load(os.path.join(full_path, generate_save_name(epoch)))
    return model


def torch_load_checkpointed(epoch, full_path, *target_modules):
    state_dict = torch.load(os.path.join(full_path, generate_checkpoint_name(epoch)))
    [eval('module.load_state_dict(state_dict)') for module in target_modules.items()]


def torch_load_deploy(epoch, full_path):
    raise NotImplementedError('this should not called in python')


class Stage(Enum):
    Train=0
    TrainEvaluate=1
    Evaluate=2
    Test=3


def evaluate_stage(args):
    '''
     @brief off_train is True and off_evaluate is False
    '''
    return args.off_train and not args.off_evaluate


def test_stage(args):
    '''
     @brief only run test
    '''
    return not args.off_test


def train_stage(args):
    '''
     @brief if off_train is False, it is in train_stage, and if the off_evaluate is False, it means the TrainEvaluate
    '''
    return not args.off_train
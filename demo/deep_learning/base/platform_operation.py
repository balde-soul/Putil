# coding=utf-8
import Putil.base.logger as plog


def deploy(args, model, epoch, path, example):
    if args.framework == 'torch':
        assert example is not None
        import torch
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(os.path.join(path, '{}-traced_model-jit.pt'.format(epoch)))
        pass
    else:
        raise NotImplementedError('deploy of framework: {} is not implemented'.format(args.framework))
    return None


def read_deploy(args, epoch, path):
    raise NotImplementedError('read_deploy is not implemented')
    pass


def checkpoint(args, model, optimizer, auto_save, auto_stop, lr_reduce, epoch, path):
    if args.framework == 'torch':
        import torch
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'auto_save': auto_save.state_dict(),
            'auto_stop': auto_stop.state_dict(),
            'lr_reduce': lr_reduce.state_dict()
        }
        torch.save(state_dict, os.path.join(path, '{}.pkl'.format(epoch)))
        pass
    else:
        raise NotImplementedError('checkpoint of framework: {} is not implemented'.format(args.framework))
    return None
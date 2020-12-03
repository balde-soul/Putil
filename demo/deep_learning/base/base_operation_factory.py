# coding=utf-8


def checkpoint_factory(args):
    pass


def save_factory(args):
    pass


def deploy_factory(args):
    pass


def get_models_factory(args):
    pass


def load_saved_factory(args):
    return eval('{}_load_saved'.format(args.framework))


def load_checkpointed_factory(args):
    return eval('{}_load_checkpoint_factory'.format(args.framework))


def load_deployed(args):
    return eval('{}_load_deployed'.format(args.framework))


def generate_model_element_factory(args):
    return eval('{}_generate_model_element'.format(args.framework))
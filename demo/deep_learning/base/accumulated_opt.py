# coding=utf-8

def accumulated_opt_common_args(parser):
    parser.add_argument('--accumulation_time', type=int, default=1, action='store', \
        help='how many time to accumulation the gradient, 本参数可以近似的提高batch而不占用过多显存，batch_accumulation表示' \
            '每多少个batch更新参数')
    pass

def DefaultAccumulatedOptArg(parser):
    accumulated_opt_common_args(parser)
    pass

class torch_DefaultAccumulatedOpt:
    def __init__(self, args):
        self._accumulation = args.accumulation_time
        self._count = 0
        pass

    def append(self, loss, optimization, force_accumulation=None):
        (loss / self._accumulation).backward() if force_accumulation is None else (loss / force_accumulation).backward()
        self._count += 1
        if self._count == (self._accumulation if force_accumulation is None else force_accumulation):
            optimization.step()
            optimization.zero_grad()
            self._count = 0
            pass
        pass
    pass
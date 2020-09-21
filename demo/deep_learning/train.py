# coding=utf-8


if __name__ == '__main__':
    import Putil.base.arg_base as pab
    import Putil.base.save_fold_base as psfb
    # the auto save TODO:
    from Putil.trainer.auto_save_args import generate_args as auto_save_args
    # the auto stop TODO:
    from Putil.trainer.auto_stop_args import generate_args as auto_stop_args
    # the lr_reduce TODO:
    from Putil.trainer.lr_reduce_args import generate_args as lr_reduce_arg
        # the default arg
    ppa = pab.ProjectArg(save_dir='./result', log_level='Info', debug_mode=True, config='')
    ## :auto stop setting
    auto_stop_args(ppa.parser)
    ## :auto save setting
    auto_save_args(ppa.parser)
    ## :lr reduce setting
    lr_reduce_args(ppa.parser)
    # debug
    parser.add_argument('--remote_debug', action='store_true', default=False, \
        help='setup with remote debug(blocked while not attached) or not')
    parser.add_argument('--frame_debug', action='store_true', default=False, \
        help='run all the process in two epoch with tiny data')
    # mode
    ppa.parser.add_argument('--train_off', action='store_true', default=False, \
        help='do not run train or not')
    ppa.parser.add_argument('--only_test', action='store_true', default=False, \
        help='only run test or not')
    # data setting
    parser.add_argument('--train_data_using_rate', action='store', type=float, default=1.0, \
        help='rate of data used in train')
    parser.add_argument('--evaluate_data_using_rate', action='store', type=float, default=1.0, \
        help='rate of data used in evaluate')
    parser.add_argument('--test_data_using_rate', action='store', type=float, default=1.0, \
        help='rate of data used in test')
    ppa.parser.add_argument('--naug', action='store_true', \
        help='do not use data aug while set')
    ppa.parser.add_argument('--fake_aug', action='store', type=int, default=0, \
        help='do the sub aug with NoOp for fake_aug time, check the generate_dataset')
    # train setting
    ppa.parser.add_argument('--epochs', type=int, default=10, metavar='N', \
        help='number of epochs to train (default: 10)')
    ppa.parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    ppa.parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', \
        help='input batch size for testing (default: 1000)')
    ppa.parser.add_argument('--log_interval', type=int, default=10, metavar='N', \
        help='how many batches to wait before logging training status(default: 10)')
    ppa.parser.add_argument('--summary_interval', type=int, default=100, metavar='N', \
        help='how many batchees to wait before save the summary(default: 100)')
    ppa.parser.add_argument('--evaluate_interval', type=int, default=1, metavar='N', \
        help='how many epoch to wait before evaluate the model(default: 1), '\
            'test the mode while the model is savd, would not run evaluate while -1')
    ppa.parser.add_argument('--compute_efficiency', action='store_true', default=False, \
        help='evaluate the efficiency in the test or not')
    ppa.parser.add_argument('--data_rate_in_compute_efficiency', type=int, default=200, metavar='N', \
        help='how many sample used in test to evaluate the efficiency(default: 200)')
    # model setting
    ppa.parser.add_argument('--weight', type=str, default='', action='store', \
        help='specify the pre-trained model path(default\'\')')
    ppa.parser.add_argument('--backbone_weight_path', type=str, default='', action='store', \
        help='specify the pre-trained model for the backbone, use while in finetune mode, '\
            'if the weight is specify, the backbone weight would be useless')
    args = ppa.parser.parse_args()
    
    # TODO: build the net
    # TODO: build the loss
    # TODO: build the optimization
    # TODO: build the train
    # TODO: build the evaluate
    # TODO: build the test
    # TODO: to_cuda
    
    if args.only_test:
        assert args.weight != '', 'should specify the pre-trained weight while run only_test, please specify the args.weight'
        state_dict = torch.load(args.weight)
        model.load_state_dict(state_dict)
        test()
    else:
        if args.weight != '':
            TrainLogger.info('load pre-trained model: {}'.format(args.weight))
            state_dict = torch.load(args.weight)
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(state_dict)
            auto_stop.load_state_dict(state_dict)
            auto_save.load_state_dict(state_dict)
            lr_reduce.load_state_dict(state_dict)
        for epoch in range(0, args.epochs):
            train_ret = train(epoch)
            global_step = (epoch + 1) * len(train_loader)
            # : run the val
            if ((epoch + 1) % args.evaluate_interval == 0) and (args.evaluate_interval != -1):
                if args.evaluate_off is False:
                    evaluate_ret = evaluate() 
                    if evaluate_ret[0] is True:
                        break
                    pass
                else:
                    TrainLogger.info('evaluate_off, do not run the evaluate')
                    pass
                pass
            pass
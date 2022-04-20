from importlib import import_module
# from dataloader import MSDataLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
# from utility import illegal

class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:    # 是否需要训练->测试集合不需要训练
            print('Please claim args.test_only')
            module_train = import_module('data.' + args.data_train.lower())     
            trainset = getattr(module_train, args.data_train)(args)    # getattr() 函数用于返回一个对象属性值

            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_threads,
                **kwargs
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'BSD100', 'Urban100']:  
            if not args.benchmark_noise:   # use noisy benchmark sets
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:                          # prepare to delete
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:
            module_test = import_module('data.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_threads,
            **kwargs
        )

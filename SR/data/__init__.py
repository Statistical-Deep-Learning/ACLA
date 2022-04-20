#
# from importlib import import_module
# #from dataloader import MSDataLoader
# from torch.utils.data import dataloader
# from torch.utils.data import ConcatDataset
#
# # This is a simple wrapper function for ConcatDataset
# class MyConcatDataset(ConcatDataset):
#     def __init__(self, datasets):
#         super(MyConcatDataset, self).__init__(datasets)
#         self.train = datasets[0].train
#
#     def set_scale(self, idx_scale):
#         for d in self.datasets:
#             if hasattr(d, 'set_scale'): d.set_scale(idx_scale)
#
#
# class Data:
#     def __init__(self, args):
#         self.loader_train = None
#
#         if not args.test_only:
#             datasets = []
#             d = args.data_train
#             module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
#             m = import_module('data.' + module_name.lower())
#             datasets.append(getattr(m, module_name)(args))
#
#             self.loader_train = dataloader.DataLoader(
#                 MyConcatDataset(datasets),
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 pin_memory=not args.cpu,
#                 num_workers=args.n_threads,
#             )
#
#         self.loader_test = []
#         d = args.data_test
#         if d in ['Set5', 'Set14', 'B100', 'Urban100']:
#             m = import_module('data.benchmark')
#             testset = getattr(m, 'Benchmark')(args, train=False)
#         else:
#             module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
#             m = import_module('data.' + module_name.lower())
#             testset = getattr(m, module_name)(args, train=False)
#
#         self.loader_test.append(
#             dataloader.DataLoader(
#                 testset,
#                 batch_size=1,
#                 shuffle=False,
#                 pin_memory=not args.cpu,
#                 num_workers=args.n_threads,
#             )
#         )
#



from importlib import import_module

# from dataloader import MSDataLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

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
        if not args.test_only:    # 是否需要训练
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)    # getattr() 函数用于返回一个对象属性值
            print('The length of trainset is ' + str(len(trainset)))
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_threads,
                **kwargs
            )


        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
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

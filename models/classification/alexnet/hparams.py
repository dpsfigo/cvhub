'''
Author: dpsfigo
Date: 2023-06-30 14:45:41
LastEditors: dpsfigo
LastEditTime: 2023-07-08 17:01:11
Description: 请填写简介
'''
class HParams:
    def __init__(self, **kwargs) -> None:
        self.data = {}
    
        for key, value in kwargs.items():
            self.data[key] = value
    
    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError(" HParams object has no attribute %s" % key)
        return self.data[key]

hparams = HParams(
    nepochs = 10,
    batch_size = 16,
    number_worker = 2,
    initial_learning_rate = 1e-4,
    checkpoint_interval = 10,
    save_optimizer_state=True,
    watch_interval = 10
)
    
    
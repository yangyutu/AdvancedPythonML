# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:42:25 2020

@author: yangy

from https://github.com/lanpa/tensorboardX
"""

from tensorboardX import SummaryWriter
import numpy as np
writer = SummaryWriter('log')

for n_iter in range(100):
    writer.add_scalar('data/dummy1', np.sin(n_iter), n_iter)

print('done')

# export scalar data to JSON for external processing
writer.export_scalars_to_json("./dummyScaler.json")
writer.close()

import json
with open('dummyScaler.json') as f:
    data = json.load(f)
    
print(data)



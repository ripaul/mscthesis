from multiprocessing import Pool

import dill

from experimental_setup import *

session = "bruteforce"
print(len(args))

parallel = True
if parallel:
    with Pool(n_parallel) as p:
        data_arr = p.starmap(bruteforce_sampling, args)
else:
    data_arr = []
    for arg in args:
        print(arg)
        data_arr.append(bruteforce_sampling(*arg))
        
with open("data/" + session + '_raw', "wb") as fhandle:
    dill.dump(data_arr, fhandle)
#    !git add data/* && git commit -m "bruteforce run"
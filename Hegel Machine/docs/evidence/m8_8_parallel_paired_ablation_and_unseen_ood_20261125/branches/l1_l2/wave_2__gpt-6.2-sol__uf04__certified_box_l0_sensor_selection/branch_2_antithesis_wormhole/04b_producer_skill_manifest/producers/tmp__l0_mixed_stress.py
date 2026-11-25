from concurrent.futures import ProcessPoolExecutor
import pickle
import numpy as np
from stress_l0 import check, inst, oracle

def run(seed):
    rng = np.random.default_rng(seed)
    sizes = (3, 2, 1) if seed < 830 else (3, 3, 2, 2, 2, 1, 1, 1)
    n = 26
    group = rng.normal(size=(n, len(sizes))) + .28*rng.normal(size=(n,1))
    signs = rng.choice([-1,1], size=sum(sizes))
    X = (np.repeat(group, sizes, axis=1) + .025*rng.normal(size=(n,sum(sizes))))*signs
    y = group @ rng.uniform(.25,1.5,len(sizes)) + .18*rng.normal(size=n)
    a = inst(X,y,(.035,.15,.3)[seed%3],(.55,.85,1.25)[seed%3],(.22,.48,.75)[seed%3])
    result = check(('mixed',seed),a,oracle(a) if seed < 830 else None)
    if result is None:
        with open(f'/tmp/l0_mixed_failure_{seed}.pkl','wb') as f:
            pickle.dump(a,f)
    return seed,result

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(run,(821,822,831,832,833)):
            print(row,flush=True)

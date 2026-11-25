from concurrent.futures import ProcessPoolExecutor
import numpy as np
from stress_box_l0 import check


def main():
    specs = []
    def add(name, X, y, ridge, box, pen, ref=None):
        specs.append((name, dict(X=X, y=y, ridge=ridge, box=box, l0_penalty=pen), ref))
    for seed in range(8):
        rng = np.random.default_rng(1500+seed)
        F, _ = np.linalg.qr(rng.normal(size=(25,5)))
        F += (.03,.18)[seed%2]*rng.normal(size=(25,1))/5
        X = np.repeat(F,3,axis=1) + (.005,.035)[seed%2]*rng.normal(size=(25,15))/5
        y = F@rng.uniform(.7,1.45,size=5) + .03*rng.normal(size=25)
        add(f'triples-{seed}',X,y,.035,(.75,1.5)[seed%2],(.12,.28,.45)[seed%3])
    for seed in range(6):
        rng = np.random.default_rng(1650+seed)
        X = rng.normal(size=(9,6)); X[:,1:3] = X[:,[0]] + 1e-4*X[:,1:3]
        y = X @ rng.normal(size=6) + .1*rng.normal(size=9)
        add(f'bound-active-oracle-{seed}',X,y,1e-6,(.1,.4)[seed%2],(.12,.4,.8)[seed%3],'oracle')
    with ProcessPoolExecutor(8) as pool:
        for r in pool.map(check,specs): print(*r,flush=True)

if __name__=='__main__': main()

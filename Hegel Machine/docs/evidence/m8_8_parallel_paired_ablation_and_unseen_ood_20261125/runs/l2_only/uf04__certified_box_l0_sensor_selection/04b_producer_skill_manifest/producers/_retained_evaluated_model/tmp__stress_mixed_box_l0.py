from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.linalg import hadamard
from stress_box_l0 import check

def main():
    specs=[]
    def add(name,X,y,ridge,pen,box=2,ref=None):
        specs.append((name,dict(X=X,y=y,ridge=ridge,box=box,l0_penalty=pen),ref))
    C=np.repeat(np.eye(8),2,axis=1)[:,:15]
    W=hadamard(16).astype(float)*.25
    add('mixed-duplicates-exact', W[:,:8]@C, W[:,:8]@np.ones(8), .02,.22,ref=8*(.5*.02/1.02+.22))
    add('mixed-orthogonal-tie', W[:,:15],W[:,:15]@np.full(15,1.5),.125,1.,ref=15*1.125)
    for seed in range(12):
        rng=np.random.default_rng(seed+230)
        U,_=np.linalg.qr(rng.normal(size=(28,8)))
        shared=rng.normal(size=(28,1)) / np.sqrt(28)
        F=U+(.05,.2,.45)[seed%3]*shared
        X=F@C+(.002,.02,.06)[seed%3]*rng.normal(size=(28,15))/np.sqrt(28)
        y=F@rng.uniform(.72,1.22,size=8)+.04*rng.normal(size=28)
        add(f'mixed-pairs-{seed}', X,y,(.02,.04,.08)[seed%3],(.15,.28,.42)[seed%3],(.85,1.5,2.)[seed%3])
    with ProcessPoolExecutor(8) as pool:
        for r in pool.map(check,specs): print(*r,flush=True)
if __name__=='__main__': main()

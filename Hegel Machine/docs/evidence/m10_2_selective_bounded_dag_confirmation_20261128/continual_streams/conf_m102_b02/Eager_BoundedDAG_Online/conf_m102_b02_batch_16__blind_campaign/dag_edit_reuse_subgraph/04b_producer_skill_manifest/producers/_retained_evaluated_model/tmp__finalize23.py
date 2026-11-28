from pathlib import Path
base=Path('/tmp/base23.py').read_text().split('\ndef run_node')[0]
search=Path('/tmp/search23.py').read_text().split('\ndef banks')[1].split("\nif __name__")[0]
search='\ndef banks'+search
search=search.replace('for penalty in [0.,2.,8.]:','for penalty in [0.,4.]:').replace("'maxiter':180", "'maxiter':145")
extra='''
def extra_groups(g,z,B,C,E):
    b,c,e=B(z),C(z),E(z)
    vals=[z,b,c,e,b-.5*z,b-z,b+z,b+c,b-c,b+e,b-e,e+c,e-c]
    for i,x in enumerate(vals):
        opp=np.roll(x,2,axis=1); nxt=np.roll(x,-1,axis=1)
        for j,v in enumerate([opp,nxt]):
            for s in [-1,1]:
                u=x+s*v
                for k,a in enumerate([.65,1.25,2.15]):
                    h=np.tanh(a*u); r=u/(1+(a*u)**2)
                    g[f'p{i},{j},{s},{k}']=np.c_[h,r,h*h,r*r]
            g[f'triple{i},{j}']=x*v*np.roll(x,j+1,axis=1)
        for k,a in enumerate([.65,1.25,2.15]):
            h=np.arcsinh(a*x); r=x/(1+(a*x)**2)
            g[f'quot{i},{k}']=np.c_[h,h*h,x*h,r,r*r]
    for i,x in enumerate(vals[:4]):
        for j,v in enumerate(vals[:4]):
            if i!=j:
                g[f'mix{i},{j}']=np.c_[x/(1+1.25*v*v),x/(1+2.15*v*v),x*np.tanh(1.25*v),x*x*v]
    return g

def predict_local(s,variant):
    z,B,C,E=coordinates(s); tr=s['train']; t=s['eval_times']; T=tr['traj'].T
    def basis(q): return -np.expm1(-t[:,None]*q)
    def err(q):
        a=basis(q); return (a@fit(a,T)-T).ravel()
    a=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)
    modes=fit(a,T).T; n=len(modes); rotation=np.linalg.svd(a,full_matrices=False)[2].T
    target=np.c_[tr['y'],modes@rotation]
    g=banks(z,B,C,E)
    if variant==2:g=extra_groups(g,z,B,C,E)
    names,K=covs(g,n)
    p=regress(K,target)
    p[:,1:]=p[:,1:]@rotation.T
    return p,modes,z,B(z)

def run_node(inputs: dict) -> dict:
    p,modes,z,b=predict_local(inputs['__input__'],VARIANT)
    n=len(modes)
    scale=np.maximum(modes.std(0),1e-5)
    return {'y_hat':p[n:,0].copy(),'trajectory_design':np.c_[p[:,1:]/scale,.035*z,.035*b]}
'''
Path('/tmp/challenger23.py').write_text(base+'\nfrom scipy.optimize import minimize\nfrom scipy.linalg import cho_factor,cho_solve\n'+search+extra+'\nVARIANT=1\n')

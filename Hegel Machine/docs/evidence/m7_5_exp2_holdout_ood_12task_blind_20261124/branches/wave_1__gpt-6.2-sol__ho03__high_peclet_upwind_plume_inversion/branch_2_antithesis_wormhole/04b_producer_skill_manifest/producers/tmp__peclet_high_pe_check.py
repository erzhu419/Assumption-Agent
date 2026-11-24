import pickle, os
from time import perf_counter
import numpy as np
from peclet_plume import forward_peclet_field
from peclet_stable import invert_point_source as stable
from peclet_stable_prior import invert_point_source as regularized
with open('/holdout_task/public_peclet.pkl','rb') as f:
    data=pickle.load(f)
p=data['problem']; n=p['N']; h=p['h']; L=n*h
x=(np.arange(n)+.5)*h
rng=np.random.default_rng(207354)
groups={k:[] for k in ('public','high_Pe_off_core','high_Pe_on_core','mixed_directions')}
groups['public'].append((p,np.array(data['gt_source_xy']),data['gt_Q']))
for group in tuple(groups)[1:]:
    for j in range(12):
        xy=L*rng.uniform([.17,.22],[.57,.55]); Q=rng.uniform(2,8); sigma=L*rng.uniform(.036,.063)
        ux,uy,ij=p['ux_face'],p['uy_face'],p['sensor_ij']
        if j%2:
            ux,uy,ij=uy.T,ux.T,ij[:,::-1]; xy=xy[::-1]
        if j%3==0:
            ux,uy,ij=-ux[::-1,:],uy[::-1,:],ij*[-1,1]+[n-1,0]; xy=np.array([L-xy[0],xy[1]])
        if group=='mixed_directions':
            ux=ux-ux.mean()*.98; uy=uy-uy.mean()*.93
        D=h*max(abs(ux).max(),abs(uy).max())/rng.uniform(60,110); kappa=rng.uniform(.12,.5)
        rhs=Q*np.exp(-.5*((x[:,None]-xy[0])**2+(x[None,:]-xy[1])**2)/sigma**2)
        c=forward_peclet_field(ux,uy,D,kappa,rhs,h)[tuple(ij.T)]
        rank=np.argsort(c); bad=rank[-2:] if group=='high_Pe_on_core' else rng.choice(rank[:-4],2,replace=False)
        noise=rng.choice([.008,.02])*c.max()
        y=c+rng.normal(0,noise,len(c)); y[bad]+=c.max()*np.array([2.7,2.2 if j%3==0 else -2.2])
        groups[group].append((dict(p,ux_face=ux,uy_face=uy,sensor_ij=ij,D=D,kappa=kappa,sigma_s=sigma,c_obs=y),xy,Q))
for label,f in [('stable',stable),('regularized',regularized)]:
    os.environ['PECLET_PRIOR_TEST']='12'
    for name,cases in groups.items():
        e=[]
        for p,xy,Q in cases:
            t=perf_counter(); r=f(p); dt=perf_counter()-t
            e.append([np.linalg.norm(np.asarray(r['source_xy'])-xy)/L,abs(r['strength_Q']/Q-1),dt])
        e=np.asarray(e); failed=(e[:,0]>.025)|(e[:,1]>.08)
        print(label,name,'failed',f'{failed.sum()}/{len(e)}','max',e.max(axis=0).round(6),'p90',np.quantile(e,.9,axis=0).round(6),flush=True)

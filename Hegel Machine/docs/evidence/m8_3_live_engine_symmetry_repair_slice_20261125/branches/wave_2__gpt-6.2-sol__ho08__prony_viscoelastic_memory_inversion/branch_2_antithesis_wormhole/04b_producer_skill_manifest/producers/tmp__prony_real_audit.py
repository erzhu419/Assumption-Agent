import sys,pickle
import numpy as np
from scipy.linalg import qr
sys.path.insert(0,'/holdout_task')
import prony_rheology as m
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']
t,x,y=(e[k] for k in ('train_t','train_strain','train_stress_raw'))
D=np.c_[np.ones(len(t)),(t-t.mean())/np.ptp(t)]
tau=np.geomspace(.02,20,96); A=m._design(t,x,tau)
z,c=m._fit(t,x,y); B=m._design(t,x,z)
keep,s=m._robust_screen(A,D,y)
r=y-B@c; b=np.linalg.lstsq(D[keep],r[keep],rcond=None)[0]; r-=D@b
print('time/strain every 2s', list(zip(t[::40],x[::40].round(4))))
print('fitted drift',b,'rejected',sum(~keep),'inlier MAD',s,'lag-1 inlier correlation',np.corrcoef(r[:-1][keep[:-1]&keep[1:]],r[1:][keep[:-1]&keep[1:]])[0,1])
print('orthogonal residual after eliminating affine drift',np.linalg.norm(D[keep].T@r[keep]))
for j,ix in enumerate(np.array_split(np.arange(len(t)),10)):
    tr=np.ones(len(t),bool); tr[max(0,ix[0]-2):ix[-1]+3]=False
    scores=[]
    for label,X in [('offset only',D[:,:1]),('affine',D)]:
        k,s0=m._robust_screen(A[tr],X[tr],y[tr]); sel=np.flatnonzero(tr)[k]
        Q=qr(X[sel],mode='economic')[0]; g=m._positive_fit(m._project(Q,A[sel]),m._project(Q,y[sel])); nuisance=np.linalg.lstsq(X[sel],y[sel]-A[sel]@g,rcond=None)[0]; rv=y[ix]-A[ix]@g-X[ix]@nuisance
        scores.append((label,round(float(np.median(abs(rv))),5),round(float(np.sqrt(np.mean(np.minimum(abs(rv),6*s0)**2))),5),round(g[0],3)))
    print('fold',j+1,'time',t[ix[0]],t[ix[-1]],scores)

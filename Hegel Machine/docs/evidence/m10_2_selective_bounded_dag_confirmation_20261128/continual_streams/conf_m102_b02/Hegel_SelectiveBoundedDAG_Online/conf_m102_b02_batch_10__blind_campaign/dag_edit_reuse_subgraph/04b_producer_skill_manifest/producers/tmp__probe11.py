import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import node_00 as n
from scipy.optimize import least_squares,minimize_scalar
from _frozen_nodes import node_01,node_02,node_03
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a']; u,v=p['u'],p['v']; keep=abs(v)<3

def fitA(w,kind):
    def f(x,j):
        a=x[:,j]; b=x[:,(j+2)%4]
        g={'sin':lambda:np.sin(w*b),'tanh':lambda:np.tanh(w*b),'rational':lambda:b/(1+w*b*b),'gauss':lambda:b*np.exp(-w*b*b)}[kind]()
        return np.c_[np.ones(len(x)),a,a**3,g]
    pars=[]; errs=[]
    for j in range(4):
        d=f(u,j); m=keep[:,j]; cf=np.linalg.lstsq(d[m],v[m,j],rcond=None)[0]; pars.append(cf); errs.extend(d[m]@cf-v[m,j])
    return np.mean(np.square(errs)),lambda x:np.column_stack([f(x,j)@pars[j] for j in range(4)]),np.array(pars)

def inverse(f,x): return np.array([least_squares(lambda q:f(q[None])[0]-row,row).x for row in x])

def cv(x,c,inds):
    it=np.setdiff1d(np.arange(len(x)),inds); coords=dict(x_train=x[it],x_test=x[inds],c_train=c[it],c_test=c[inds]); st=dict(s,train={k:s['train'][k][it] for k in ('x','y','traj')}); d=node_01.run_node({'node_00':coords}); sup=node_02.run_node({'__input__':st}); out=node_03.run_node(dict(node_00=coords,node_01=d,node_02=sup)); return [np.mean(abs(out[k]-s['train'][j][inds])) for k,j in [('y_hat','y'),('state_pred','traj')]]
A={}; splits=[np.arange(52,72),np.arange(0,20),np.arange(26,46)]
for kind in ['sin','tanh','rational','gauss']:
    opt=minimize_scalar(lambda w:fitA(w,kind)[0],bounds=(.1,5),method='bounded'); e,f,co=fitA(opt.x,kind); x=inverse(f,s['train']['x']); A[kind]=x; print('A',kind,opt.x,e**.5,np.round(co,5), 'roundtrip',abs(f(x)-s['train']['x']).max()); c=n._panel_c(s['panel_c'],n._panel_b(s['panel_b'],x)); print('cv',np.round([cv(x,c,i) for i in splits],5))
C=s['panel_c']; u=C['u']; v=C['v1']; print('C singular vals',np.linalg.svd(v,compute_uv=False));
def fitC(f):
 d=np.c_[np.ones(len(u)),f(u)]; z=np.linalg.lstsq(d,v,rcond=None)[0]; h=np.einsum('ij,ji->i',d,np.linalg.pinv(d)); return np.sqrt(np.mean(((d@z-v)/(1-h[:,None]))**2)), lambda b:np.c_[np.ones(len(b)),f(b)]@z
for nam,fn in [('sin',np.sin),('cos',np.cos),('sin+cos',lambda w:np.c_[np.sin(w),np.cos(w)]),('tanh',np.tanh)]:
 a=sorted((fitC(lambda q:fn(w*q))[0],w) for w in np.linspace(.1,30,400)); print('C scan',nam,a[:4]);
 for _,w in a[:1]:
  e,f=fitC(lambda q:fn(w*q)); x=A['sin']; b=n._panel_b(s['panel_b'],x); print('C cv',nam,np.round([cv(x,f(b),i) for i in splits],5))
print('C rows',np.round(np.c_[u,v],4).tolist())

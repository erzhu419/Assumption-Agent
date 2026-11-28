import pickle,numpy as np
from scipy.optimize import minimize_scalar
from importlib.machinery import SourceFileLoader
from operators import poly
from latent import reverse_a

def recover(u,v,kind):
    def design(x,w,order):
        a=poly(x,order)
        f={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'rat':lambda x:1/(1+x*x)}[kind]
        return np.c_[a,f(w*x)]
    def loo(w,order):
        A=design(u,w,order); q,r=np.linalg.qr(A); err=v-q@(q.T@v)
        return np.mean((err/(1-(q*q).sum(1,keepdims=True)))**2)
    opts=[]
    for order in [1,2]:
        ws=np.linspace(.2,4.5,80); i=np.argmin([loo(w,order) for w in ws]); lo,hi=ws[max(0,i-1)],ws[min(len(ws)-1,i+1)]
        opt=minimize_scalar(lambda w:loo(w,order),bounds=(lo,hi),method='bounded'); w=opt.x
        D=design(u,w,order); coef=np.linalg.lstsq(D,v,rcond=None)[0]; print(kind,order,round(w,5),np.sqrt(opt.fun),'coefs',coef.T.round(3) if opt.fun<1e-6 else '',flush=True)
        opts.append((opt.fun,lambda x,w=w,order=order,coef=coef:design(x,w,order)@coef))
    return min(opts,key=lambda t:t[0])

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=reverse_a(s['panel_a'],tr['x']); p=s['panel_b']; bb=np.linalg.lstsq(poly(p['u'],1),p['v'],rcond=None)[0]; p=s['panel_c']; c=[]
 for k,ks in [('v1',['tanh','sin']),('v2',['cos','rat'])]:
    c.append(min([recover(p['u'],p[k],kind) for kind in ks],key=lambda t:t[0])[1])
 prior=SourceFileLoader('prior','/tmp/hegel_continual_archive/conf_m101_b03_batch_00_solver.py').load_module()
 for tag,w in [('z',z),('Bz',poly(z,1)@bb),('iBz',(z-bb[0])@np.linalg.inv(bb[1:]))]:
    v1,v2=[f(w) for f in c]; D0=poly(w,1); D2=poly(w,2)
    for name,D in [('C1',np.c_[D0,v1]),('C2',np.c_[D0,v2]),('C12',np.c_[D0,v1,v2]),('C12Q',np.c_[D2,v1,v2]),('C12prod',np.c_[D0,v1,v2,v1*v2,v1*v1,v2*v2]),('C12QB',np.c_[D2,v1,v2,w*v1,w*v2,v1*v2])]:
      scores=[]
      for n in [52,56]:
        H,M=prior._time_modes(s['eval_times'],tr['traj'][:n]); Y=np.c_[tr['y'][:n],M]; pred=D[n:]@np.linalg.lstsq(D[:n],Y,rcond=None)[0]; scores.append([np.mean(abs(pred[:,0]-tr['y'][n:])),np.mean(abs(pred[:,1:]@H.T-tr['traj'][n:]))]);
      print(tag,name,np.round(scores,5).tolist())

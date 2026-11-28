import pickle
import numpy as np
from itertools import combinations
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']

def aff(x): return np.column_stack((np.ones(len(x)),x))
def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def adj(z): return z*np.roll(z,-1,axis=1)
def pair(z): return np.column_stack([z[:,i]*z[:,j] for i,j in combinations(range(z.shape[1]),2)])

def panels(s,x):
    u,v=(s['panel_a'][k] for k in ('u','v')); A=aff(u); b=ls(A,v)
    for _ in range(12):
        r=np.linalg.norm(v-A@b,axis=1); w=(1+(r/max(3*np.median(r),1e-6))**2)**-.5
        b=ls(w[:,None]*A,w[:,None]*v)
    r=np.linalg.norm(v-A@b,axis=1); mask=r<max(8*np.median(r),1e-4); b=ls(A[mask],v[mask]); z=np.linalg.solve(b[1:].T,(x-b[0]).T).T
    u=s['panel_c']['u']; c=aff(adj(z))@ls(aff(adj(u)),s['panel_c']['v1']); d=aff(np.cos(2.4*z))@ls(aff(np.cos(2.4*u)),s['panel_c']['v2']); return z,c,d,r

def bases(z):
    return {'linear':z,'adj':adj(z),'linear+adj':np.c_[z,adj(z)],'pair':pair(z),'quadratic':np.c_[z,pair(z),z*z],'cos':np.cos(2.4*z),'sin':np.sin(2.4*z),'tanh':np.tanh(z),'cube':z**3,'triple':np.column_stack([np.prod(z[:,ix],axis=1) for ix in combinations(range(4),3)])}

def scan_panel(s):
    for p,k in [('panel_b','v'),('panel_c','v1')]:
        u=s[p]['u']; y=s[p][k]; bb=bases(u); bb.update({f'lin+{n}':np.c_[u,f] for n,f in list(bb.items()) if n in ('cos','sin','tanh','cube','triple')}); print(p,k,[(n,round(np.mean(abs(aff(f)@ls(aff(f),y)-y)),7)) for n,f in bb.items()]);
        for freq in [0.75,1,1.2,1.5,1.7,1.8,2,2.2,2.4,2.6,3]:
            print(freq,end=':');
            for f in [np.sin(freq*u), np.cos(freq*u),np.tanh(freq*u),np.c_[np.sin(freq*u),np.cos(freq*u)]]:
                a=aff(f); print('%.5f'%np.mean(abs(a@ls(a,y)-y)),end=' ')
            print()

def cv(f,y,lam=0):
    f=np.asarray(f); errs=[]
    for i in [np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,52)]:
        j=np.setdiff1d(np.arange(72),i); a=f[j]; m=a.mean(0); sd=np.maximum(a.std(0),1e-8); a=aff((a-m)/sd); at=aff((f[i]-m)/sd)
        if lam:
            d=np.eye(a.shape[1])*lam; d[0,0]=0; v=np.linalg.solve(a.T@a+d,a.T@y[j]);
        else: v=ls(a,y[j]);
        errs.append(np.mean(abs(at@v-y[i]),axis=0))
    return np.array(errs)

if __name__=='__main__':
    scan_panel(S)
    X=S['train']['x']; Y=S['train']['y']; T=S['train']['traj']; z,c,d,r=panels(S,X); print('A robust norms',np.sort(r)[[0,22,44,45,46,47]]); print('T svd',np.linalg.svd(T,compute_uv=False));
    fd=bases(z); fd.update({'raw':X,'c':c,'d':d,'cd':np.c_[c,d], 'zcd':np.c_[z,c,d], 'zcd+z*c':np.c_[z,c,d,z*c], 'zcd+z*d':np.c_[z,c,d,z*d], 'zcd+c*d':np.c_[z,c,d,c*d], 'zcd+all3':np.c_[z,c,d,z*c,z*d,c*d,z*c*d], 'zcd+outerzc':np.c_[z,c,d,(z[:,:,None]*c[:,None,:]).reshape(len(z),-1)], 'zcd+pairz':np.c_[z,c,d,pair(z),z*z], 'zcd+polycd':np.c_[z,c,d,z*z,c*c,d*d,pair(c),pair(d)], 'z+triple':np.c_[z,adj(z),np.roll(z,1,axis=1)*adj(z)]});
    for name,f in fd.items():
        ey=cv(f,Y); et=cv(f,T); print(name,f.shape[1], 'y',np.round(ey,4),'traj',np.round(et.mean(1),4),'traj last',np.round(et[:,-1],4));

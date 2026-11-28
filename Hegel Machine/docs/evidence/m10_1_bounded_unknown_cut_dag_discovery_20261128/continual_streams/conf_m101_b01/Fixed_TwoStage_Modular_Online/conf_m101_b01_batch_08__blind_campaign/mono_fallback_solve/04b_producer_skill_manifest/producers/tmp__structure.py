from compose import *
from scipy.special import expit
u=s['panel_b']['u']; v=s['panel_b']['v']; a=afit(s)['diagsecond']; z=inverse(a,s['train']['x']); op=operators(s)
def inspect_b():
    n=len(u); R2=lambda x:np.roll(x,-2,axis=1); h=lambda x:x-.4*np.sin(2.8*x)
    for xname,x in [('u',u),('hu',h(u))]:
        r=R(x); q=R2(x)
        fs={'quad':mono(x,2),'cube':x**3,'cycle':x*r,'cubeCross':x*x*r,'cycle3':x*r*q,'fullcube':mono(x,3),'pair sin':np.sin(2.8*(x+r)),'pair cos':np.cos(2.8*(x+r)),'prod sin':np.sin(2.8*x*r),'prod cos':np.cos(2.8*x*r),'difference':np.sin(2.8*(x-r)),'axis sin2':np.sin(5.6*x),'axis cos':np.cos(2.8*x),'axis tanh':np.tanh(x),'anti':(x*q)[:,:2],'rat':x/(1+x*x),'self*sin':x*np.sin(2.8*x),'neighbor*sin':r*np.sin(2.8*x)}
        X=aff(h(u)); w=ls(X,v); e=v-X@w
        print('B residual',xname,'singular',np.linalg.svd(e,compute_uv=False),'max',np.max(abs(e),axis=0))
        results=[]
        for name,d in fs.items():
            A=np.c_[X,d]; H=A@np.linalg.pinv(A); E=(v-H@v)/np.maximum(1-np.diag(H),1e-6)[:,None]; results.append((np.mean(abs(E)),name,np.mean(abs(v-H@v))))
        print(sorted(results)[:14]); print('inverse residual',e[:6]@np.linalg.inv(w[1:]))
inspect_b()
Z={'Z':z}
for k in ('B','C','D','BC','BD','CB','CD','DB','DC','BCD','BDC','CBD','CDB','DBC','DCB'):
    Z[k]=op[k[-1]](Z[k[:-1] or 'Z'])
y=s['train']['y']; ts=s['train']['traj']; t=s['eval_times']; Y=np.c_[y,ts]; print('trajectory SVD',np.linalg.svd(ts-ts.mean(0),compute_uv=False)); print('repeated fits to time:')
for names,T in [('poly',np.c_[np.ones(8),t,t*t]),('exp',np.c_[np.ones(8),np.exp(-t),np.exp(-2*t)]),('exp_sin',np.c_[np.ones(8),1-np.exp(-t),np.exp(-t)*np.sin(1.7*t),np.exp(-t)*np.cos(1.7*t)]),('exp_tanh',np.c_[np.ones(8),1-np.exp(-t),np.exp(-t)*np.tanh(1.7*t)])]:
 print(names,np.mean(abs(ts-(T@ls(T,ts.T)).T)),np.mean(abs(ts.mean(0)-T@ls(T,ts.mean(0)))))
print('target singular cross',np.corrcoef(Y.T)[0]); print('first rows',Y[:7])
folds=[np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(6,10)]
def cv(X,reg=.18):
    scores=[]
    for te in folds:
        tr=np.setdiff1d(np.arange(72),te); m=X[tr].mean(0); sc=X[tr].std(0).clip(1e-8); A=(X-m)/sc; b=Y[tr].mean(0); w=np.linalg.solve(A[tr].T@A[tr]+reg*np.eye(A.shape[1]),A[tr].T@(Y[tr]-b)); E=abs(A[te]@w+b-Y[te]); scores.append([E[:,0].mean(),E[:,1:].mean()])
    return np.array(scores)
res=[]
for k in ('B','BC','BCD','BD','Z'):
    x=Z[k]; r=R(x); p=x*r
    fs={'lin':x,'quad':poly(x)[:,1:],'cyc':np.c_[x,p],'branch':np.c_[x,Z['BC'],Z['BD'],Z['BCD']],'cycR':np.c_[x,p/(1+(x-r)**2)],'BCmixed':np.c_[x,p,Z['BC'],Z['BCD'],Z['BD']],'common':np.c_[x,p,np.tanh(3*x),np.tanh(3*p)]}
    for c in [.5,1,2,4,8]:
        for kn,d in [('sin',np.sin(c*p)),('tanh',np.tanh(c*p)),('atan',np.arctan(c*p)),('sats',p/np.sqrt(1+c*p*p)),('rat',p/(1+c*p*p)),('rdif',p/(1+c*(x-r)**2)),('rsum',p/(1+c*(x+r)**2)),('axrat',x/(1+c*r*r)),('tanhx',np.tanh(c*x))]:
            fs[kn+str(c)]=np.c_[x,d]; fs[kn+str(c)+'+cyc']=np.c_[x,p,d]
    for n,d in fs.items():
        E=cv(d); res.append((k,n,E))
for j in [0,1]:
 print('rank',j,[(k,n,e[0].round(4).tolist(),e.mean(0).round(4).tolist(),e.max(0).round(4).tolist()) for k,n,e in sorted(res,key=lambda a:a[2].mean(0)[j])[:14]])

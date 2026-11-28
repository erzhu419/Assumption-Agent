from model14 import *
import pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u=s['panel_c']['u'];v=s['panel_c']['v1']; matches=[]
func={**fun,'frac':lambda x:x/np.sqrt(1+x*x),'exp':lambda x:-np.expm1(-x*x)*x,'sinc':lambda x:np.sinc(x/np.pi)*x};
for nm,f in func.items():
 if nm in ('sq','cube','cos'):continue
 for gnm,g in {'cube':lambda x:x**3,'rat':lambda x:x/(1+.22*x*x),'cubrat':lambda x:x**3/(1+.22*x*x),'adj':lambda x:x*np.roll(x,1,axis=1),'none':lambda x:x[:,:0]}.items():
  for mode in (0,1,2,3):
   def dm(x,k):
    a=f(k[0]*x); b=g(x)
    if mode==0:return D(np.c_[x,a,b]);
    if mode==1 and b.size:return D(np.c_[x,a+k[1]*b]);
    if mode==2 and b.size:return D(np.c_[a+k[1]*x,b]);
    if mode==3 and b.size:return D(np.c_[a,x+k[1]*b]);
    return D(np.c_[x,a]);
   def err(k):
    d=dm(u,k);return (d@ls(d,v)-v).ravel()
   for k0 in (.6,1.5,2.15):
    fit=least_squares(err,[k0,.22],bounds=([.08,-10],[5,10]),max_nfev=70);a=dm(u,fit.x); e=fit.fun.reshape(v.shape)/np.maximum(1-np.diag(a@np.linalg.pinv(a)),.005)[:,None];matches.append((np.std(e),nm,gnm,mode,fit.x))
for m in sorted(matches,key=lambda t:t[0])[:20]:print(m)
f=np.load('/tmp/graph14.npz');p,q,Y=(f[k] for k in ('p','q','Y')); n=52
for i in range(3):
 for name,d in [('pq',D(np.c_[p,q,p*q])),('cross',D(np.c_[p,q,(p[:,:,None]*q[:,None,:]).reshape(len(p),16)])),('fast',D(np.c_[p-q,np.tanh(.22*(p+q)),p*q/(1+.22**2*(p*q)**2)]))]:
  c=ls(d[:n],Y[:n,i]);print(i,name,np.mean(abs(d[n:72]@c-Y[n:,i])),np.round(c,3))

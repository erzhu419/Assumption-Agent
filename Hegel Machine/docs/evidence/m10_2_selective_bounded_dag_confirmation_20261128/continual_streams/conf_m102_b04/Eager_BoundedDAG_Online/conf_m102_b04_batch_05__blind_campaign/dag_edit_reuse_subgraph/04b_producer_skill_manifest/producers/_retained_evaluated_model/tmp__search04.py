exec(open('/tmp/probe04.py').read().split('for p in')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
ix=np.arange(n); folds=np.array_split(np.random.default_rng(77).permutation(n),4)+[ix[-20:]]
def evaluate(kernel, regs=(.0001,.001,.007,.03,.1)):
 ys=[]; zs=[]
 for te in folds:
  tr=np.setdiff1d(ix,te); l,v=eigh(kernel[np.ix_(tr,tr)]); lhs=(kernel[np.ix_(te,tr)]@v); m=np.linalg.svd(z[tr],full_matrices=False)[2][:2]; rhs=v.T@np.c_[y[tr], (z[tr]@m.T)@m]; p=np.einsum('ij,rj,jk->rik',lhs,1/(l[None,:]+np.asarray(regs)[:,None]),rhs)
  ys.append(np.abs(p[:,:,0]-y[te]).mean(axis=1)); zs.append(np.abs(p[:,:,1:]-z[te]).mean(axis=(1,2)))
 a,b=np.array(ys),np.array(zs); return [(round(max(e.max(axis=0)[i],0),5),round(e[-1,i],5),round(e[:-1,i].mean(),5),regs[i]) for e in (a,b) for i in [np.argmin(e.max(axis=0))]]
f=np.c_[np.ones(48),s['panel_a']['u']]; v=s['panel_a']['v']; keep=np.arange(len(v));
for _ in range(10):
 fit=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0]; keep=np.argsort(np.linalg.norm(v-f@fit,axis=1))[:39]
qc=(x-fit[0])@np.linalg.inv(fit[1:]); b=np.linalg.lstsq(poly(s['panel_b']['u'],1),s['panel_b']['v'],rcond=None)[0]; c=np.linalg.lstsq(poly(s['panel_c']['u'],1),s['panel_c']['v1'],rcond=None)[0]; U,S,V=np.linalg.svd(c[1:]); c[1:]=U@V
coords={'q':q,'x':x,'fullA':qc,'B_x':poly(x,1)@b,'B_q':poly(q,1)@b,'B_fullA':poly(qc,1)@b,'C_x':poly(x,1)@c,'C_q':poly(q,1)@c,'C_fullA':poly(qc,1)@c}
res=[]
for nm,u in coords.items():
 for h in (0,.22,.45,.7,1.,1.25,1.65):
  for typ in ('sqrt','tanh','asinh') if h else ('sqrt',):
   w={'sqrt':lambda:u/np.sqrt(1+h*u*u),'tanh':lambda:np.tanh(np.sqrt(h)*u)/np.sqrt(h),'asinh':lambda:np.arcsinh(np.sqrt(h)*u)/np.sqrt(h)}[typ]()
   for d in (2,3,4,5):
    g=poly(w,d); k=g@g.T; r=evaluate(k); res.append((nm,h,typ,d,*r))
   dist=cdist(w,w,'sqeuclidean')
   for lam in (.15,.5,1.):
    r=evaluate(np.exp(-lam*dist)); res.append((nm,h,typ,'rbf'+str(lam),*r))
for j in (-2,-1):
 print('HEAD',j,flush=True)
 for row in sorted(res,key=lambda p:p[j])[:16]:print(row,flush=True)

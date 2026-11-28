exec(open('/tmp/readout_probe.py').read().split('for name,p in')[0])
from scipy.optimize import least_squares
lat=np.cbrt(q); raw=s['train']['x']; tar=np.c_[y,t[:,0],t[:,-1]]; tar=tar/tar.std(0)
ca=s['panel_a']; u,v=ca['u'],ca['v']; keep=np.linalg.norm(v-u,axis=1)<3
k=np.arange(4); acoef=np.stack([np.linalg.lstsq(np.c_[aff(u),u[:,i]**3,u[:,i]*u[:,(i+1)%4]][keep],v[keep,i],rcond=None)[0] for i in k],1)
a=np.array([acoef[0]+p@acoef[1:5]+acoef[-2]*p**3+acoef[-1]*p*p[(k+1)%4] for p in u]); print('A residual quantiles',np.quantile(np.linalg.norm(a-v,axis=1),[0,.5,.8,.9,1]),'coeff',acoef,flush=True)
c=s['panel_c']; uc=c['u']; z0=uc
for name,fn in [('linear+cos',lambda x,g:np.c_[aff(x),np.cos(g*x)]),('linear+square+cos',lambda x,g:np.c_[aff(x),x*x,np.cos(g*x)]),('linear+cubic+cos',lambda x,g:np.c_[aff(x),x**3,np.cos(g*x)]),('atan+cubic',lambda x,g:np.c_[aff(np.arctan(g*x)),x**3]),('atan+square',lambda x,g:np.c_[aff(np.arctan(g*x)),x*x]),('atan+rat',lambda x,g:np.c_[aff(np.arctan(g*x)),x/(1+g*x*x)])]:
 for key in ['v1','v2']:
  def er(g):
   a=fn(uc,g); return np.mean((a@np.linalg.lstsq(a,c[key],rcond=None)[0]-c[key])**2)**.5
  gs=np.linspace(.25,6,80); j=np.argmin([er(g) for g in gs]); o=minimize_scalar(er,bounds=(gs[max(0,j-1)],gs[min(79,j+1)]),method='bounded'); print(name,key,o.x,o.fun,flush=True)

def profile(x, mode, p0, rng):
    def feat(x,p):
        P=p[:16].reshape(4,4); h=x@P+p[16:20]
        if len(p)>20: h+=x**3*p[20:24]+x*np.roll(x,-1,axis=1)*p[24:28]
        if mode=='both':return np.c_[aff(np.cos(h)),np.sin(h)]
        if mode=='tanh2':return np.c_[aff(np.tanh(h)**2),np.tanh(h)]
        return aff(np.cos(h))
    tr,te=rng[:54],rng[54:]
    def res(p):
        a=feat(x[tr],p); return (a@np.linalg.lstsq(a,tar[tr],rcond=1e-9)[0]-tar[tr]).ravel()
    opt=least_squares(res,p0,max_nfev=180); a=feat(x,opt.x); w=np.linalg.lstsq(a[tr],tar[tr],rcond=1e-9)[0]; v=np.mean(abs(a[te]@w-tar[te]),axis=0); return v.mean(),v,np.mean(abs(opt.fun)),opt.x
rng=np.random.default_rng(142); ii=rng.permutation(n); results=[]
for name,x in [('lat',lat),('raw',raw),('cubic',q),('B',aff(np.cos(f*lat))@co),('C1',aff(np.arctan(2.15*lat))@np.linalg.lstsq(aff(np.arctan(2.15*uc)),c['v1'],rcond=None)[0])]:
 for mode in ['cos','both','tanh2']:
  for extra in [0,8]:
   for g in [1.3,2.15,2.6,3.3]:
    starts=[g*np.eye(4),g*np.eye(4)+rng.normal(0,.24,(4,4))]
    for m in starts:
     r=profile(x,mode,np.r_[m.ravel(),np.zeros(4+extra)],ii); results.append((*r[:3],name,mode,extra,r[3]));
   a=min([r for r in results if r[3:6]==(name,mode,extra)],key=lambda r:r[0]); print('RIDGE',a,flush=True)
print('BEST',*sorted(results,key=lambda a:a[0])[:4],sep='\n')

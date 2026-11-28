exec(open('/tmp/probe_rewire.py').read().split('for key in')[0])
from sklearn.linear_model import OrthogonalMatchingPursuit, lars_path
from scipy.linalg import svd
p=s['panel_c']; u=p['u']; variants={'tanh':np.tanh,'atan':np.arctan,'soft':lambda x:x/(1+abs(x)),'alg':lambda x:x/np.sqrt(1+x*x),'erf':__import__('scipy').special.erf,'sin':np.sin}
for nm,fn in variants.items():
 def obj(k):
  h=aff(fn(k*u));return np.mean((h@fit(h,p['v1'])-p['v1'])**2)
 r=minimize_scalar(obj,bounds=(.01,4),method='bounded');print(nm,r.x,r.fun**.5)
q=-np.expm1(-t[:,None]*[.4125,1.80995]); A=np.c_[Y[:,0],fit(q,Y[:,1:].T).T]; C1=fit(aff(np.arctan(2.13*u)),p['v1']); C2=fit(aff(np.cos(2.4*u)),p['v2']);
coords={'z':z,'b':aff(z)@R,'bi':(z-R[0])@np.linalg.inv(R[1:]),'c1':aff(np.arctan(2.13*z))@C1,'c2':aff(np.cos(2.4*z))@C2,'raw':X}
def lib(w):
 out={};
 def add(name,a):out[name]=a
 for i,x in enumerate(w.T):
  for k in range(1,5):add(f'x{i}^{k}',x**k)
  for k in [0.5,0.75,1.,1.25,1.5,1.75,2.,2.25,2.4,2.5,3.,3.5,4.,5.,6.]:
   for nm,fn in {**variants,'cos':np.cos,'gauss':lambda a:np.exp(-a*a)}.items():add(f'{nm}{k}(x{i})',fn(k*x))
 for i,j in combinations(range(4),2):
  x,y=w[:,i],w[:,j];
  for a,b in [(1,1),(1,2),(2,1),(2,2),(3,1),(1,3)]:add(f'x{i}^{a}x{j}^{b}',x**a*y**b)
  for k in [.25,.5,1,1.5,2]:add(f'rat{k}({i},{j})',x*y/(1+k*(x*x+y*y)))
  for c in [-1.5,-1,-.7,-.5,.5,.7,1,1.5]:
   for k in [.65,1,1.5,2,2.4,3,4]:
    for nm,fn in [('sin',np.sin),('cos',np.cos),('sat',np.tanh),('gauss',lambda a:np.exp(-a*a))]:add(f'{nm}{k}(x{i}+{c}x{j})',fn(k*(x+c*y)))
 for ix in combinations(range(4),3):add(str(ix),np.prod(w[:,ix],axis=1))
 return out
D={name+':'+k:v for name,w in coords.items() for k,v in lib(w).items()}; names=np.array(list(D)); F=np.column_stack(list(D.values())); n=52
for maskname in list(coords)+['all']:
 mask=np.char.startswith(names,maskname+':') if maskname!='all' else np.ones(len(names),bool); f=F[:,mask]; nm=names[mask]; f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-10)
 for j in range(3):
  target=A[:,j]; y=target[:n]-target[:n].mean(); al,ids,coef=lars_path(f[:n],y,method='lasso',max_iter=160); results=[]
  for k in range(1,coef.shape[1]):
   inds=np.flatnonzero(abs(coef[:,k])>1e-8)
   if len(inds)>33:continue
   g=aff(f[:,inds]); b=fit(g[:n],target[:n]); err=np.mean(abs(g[n:]@b-target[n:])); results.append((err,k,len(inds),np.sqrt(np.mean((g[:n]@b-target[:n])**2)),inds,b))
  r=min(results,key=lambda a:a[0]); print(maskname,j,np.round(r[:4],5),list(zip(nm[r[4]][:16],r[5][1:17].round(3))))

exec(open('/tmp/graph06.py').read().split('R=[]')[0].split('for shift in range')[0]+open('/tmp/graph06.py').read().split("x=s['train']['x']")[1].split('R=[]')[0].join(['\nx=s[\'train\'][\'x\']','']))
from sklearn.linear_model import Lasso,MultiTaskLasso
from itertools import product

def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
U,S,V=np.linalg.svd(Y[:52,1:],full_matrices=False); Q=V[:3].T; YY=np.c_[Y[:,0],Y[:,1:]@Q]; spread=YY[:52].std(0); A=YY/spread

def metrics(P):
 e=np.c_[P[:,0]-Y[:,0], P[:,1:]@Q.T-Y[:,1:]]; return np.round([np.abs(e[52:,0]).mean(),np.abs(e[52:,1:]).mean(),np.abs(e[:52,0]).mean(),np.abs(e[:52,1:]).mean()],5)
C=np.cos(2.4*z); J=pterms(z)
res=[]
for k,a in [('z',z),('b',b),('x',x)]:
 base=np.c_[a,g,h]; blocks={'ag':cross(a,g),'ah':cross(a,h),'gh':cross(g,h),'aa':poly(a,2)[:,5:],'gg':poly(g,2)[:,5:],'hh':poly(h,2)[:,5:],'dg':a*g,'dh':a*h,'ds':g*h,'tg':np.sum(a*g,1)[:,None],'th':np.sum(a*h,1)[:,None],'ts':np.sum(g*h,1)[:,None]}
 for j in range(0,4):
  for keys in combinations(blocks,j):
   f=bias(np.column_stack([base]+[blocks[key] for key in keys]));
   if f.shape[1]>48:continue
   P=f@ls(f[:52],YY[:52]); e=metrics(P); res.append((*e,k,keys))
for j in range(2):
 print('groups',j, *sorted(res,key=lambda r:r[j])[:12],sep='\n')
for nm,a,j,c in [('mapped',z,g,h),('bmap',b,g,h),('unmapped',z,J,C),('bmix',b,J,C)]:
 B={'z':a,'g':j,'h':c,'zg':cross(a,j),'zh':cross(a,c),'gh':cross(j,c),'zdotg':np.sum(a*j,1)[:,None],'zdoth':np.sum(a*c,1)[:,None],'gdot':np.sum(j*c,1)[:,None]};F=np.column_stack(list(B.values())); names=[f'{k}{i}' for k,v in B.items() for i in range(v.shape[1])]; sd=F[:52].std(0); f=F/sd
 print('sparse',nm)
 for alpha in [.001,.003,.01,.02,.04,.08]:
  fit=MultiTaskLasso(alpha=alpha,max_iter=12000,tol=1e-8).fit(f[:52],A[:52]); idx=np.linalg.norm(fit.coef_,axis=0)>1e-8; ff=bias(f[:,idx]); e=metrics(ff@ls(ff[:52],YY[:52])); print(alpha,idx.sum(),e,','.join(np.array(names)[idx]) if max(e[:2])<.09 else '')

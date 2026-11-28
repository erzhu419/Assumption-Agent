import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _coords,_panel_b,_poly,_ls,_roll
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=_coords(s,tr['x']);h=_panel_b(s,z); Y=np.c_[tr['y'],tr['traj']];n=52; I=np.eye(n); u=s['panel_b']['u'];v=s['panel_b']['v']; hv=_panel_b(s,u); b=_ls(np.c_[np.ones(len(u)),u],v)[0][0]; h-=b; hv-=b
pairs=list(combinations(range(4),2)); pr=lambda x:np.c_[[x[:,i]*x[:,j] for i,j in pairs]].T
aux={};exec(open('/tmp/graph7.py').read().split('B=auxiliary')[0],aux)
c1=aux['auxiliary']('panel_c','v1'); c2=aux['auxiliary']('panel_c','v2'); print('panel_B_residual',np.std(hv+b-v));
features={};
def add(k,a):
 a=np.asarray(a).reshape(len(z),-1); a-=a[:n].mean(0); a/=np.maximum(np.sqrt(np.mean(a[:n]**2,axis=0)),1e-12);features[k]=a.copy()
def expand(k,a):
 add(k+'L',a.copy()); add(k+'Q',a*a); add(k+'X',pr(a))
expand('H',h);expand('Z',z); H=h.copy(); Z=z.copy();
for k,a in [('H',H),('Z',Z)]:
 F=_poly(a,4); add(k+'D3',F[:,15:35]);add(k+'D4',F[:,35:70]);
add('HZd',H*Z);add('HZx',np.column_stack([H[:,i]*Z[:,j] for i in range(4) for j in range(4) if i!=j]));
# Use regularized tensor products in a native coordinate basis that does not inherit arbitrary panel mixing.
res=H-_ls(np.c_[np.ones(n),Z[:n]],H[:n])[0][0]-Z@_ls(np.c_[np.ones(n),Z[:n]],H[:n])[0][1:]; R=res@np.linalg.svd(res[:n],full_matrices=False)[2].T; expand('R',R);add('RZ', (R[:,:,None]*Z[:,None,:]).reshape(len(z),16));
for key,a in [('C',c1(Z)),('D',c2(Z)),('E',c1(H+b)),('F',c2(H+b))]:expand(key,a)
for key,a in [('g',Z**3/(1+.65*Z*Z)),('j',Z**3/(1+.22*Z*Z)),('p',Z-np.tanh(1.25*Z)),('q',Z-.4*np.sin(2.85*Z))]:
 expand(key,a); add(key+'cross',(a[:,:,None]*Z[:,None,:]).reshape(len(z),16));
add('ztrip',np.column_stack([np.prod(Z[:,i],axis=1) for i in combinations(range(4),3)])); add('htrip',np.column_stack([np.prod(H[:,i],axis=1) for i in combinations(range(4),3)]));
np.savez('/tmp/features12.npz',**features)
keys=list(features); K=np.array([a@a[:n].T/a.shape[1] for a in features.values()]); G=K[:,:n].copy(); K0=np.ones((len(z),n));
for metric in ['mle','loo','ard']:
 for col in [0,8]:
  y=Y[:n,col]; inds=np.arange(len(K));
  if metric=='ard':
   keys2=[k for k in keys if k in ['HL','HQ','HX','ZL','ZQ','ZX','HZd','HZx','RL','RQ','RX','RZ','CL','DL','EL','FL','gL','gQ','gX','gcross','ztrip','htrip']];
   names=[]; ms=[]
   for k in keys2:
    for j,a in enumerate(features[k].T):ms.append(np.outer(a,a[:n]));names.append(k+str(j))
   J=np.array(ms); GG=J[:,:n];
  else:J,GG,names=K,G,keys
  def calc(p):
   w=np.exp(p); A=(w[:-1]@GG.reshape(len(GG),-1)).reshape(n,n)+K0[:n]+w[-1]*I; cf=cho_factor(A); al=cho_solve(cf,y); ai=cho_solve(cf,I); d=np.diag(ai)
   if metric=='loo':
    e=al/d; f=np.mean(e*e); M=-(np.outer(ai@(2*e/d/n),al))+(ai*(2*e*e/d/n))@ai
   else:
    f=(y@al+2*np.log(np.diag(cf[0])).sum())/n;M=(ai-np.outer(al,al))/n
   der=np.r_[GG.reshape(len(GG),-1)@M.ravel(),np.trace(M)]*w
   return f,der
  fits=[]
  for seedval in [0,-3,-6]:
   p=np.full(len(J)+1,-5.+seedval);p[-1]=-5.;p[:3]=-2.;r=minimize(calc,p,jac=True,method='L-BFGS-B',bounds=[(-17,4)]*len(J)+[(-15,0)],options={'maxiter':500,'ftol':1e-11}); w=np.exp(r.x); A=(w[:-1]@J.reshape(len(J),-1)).reshape(len(z),n)+K0; cf=cho_factor(A[:n]+w[-1]*I);al=cho_solve(cf,y);pred=A[n:]@al;loo=al/np.diag(cho_solve(cf,I)); e=np.mean(abs(pred-Y[n:,col]));fits.append((r.fun,e,w,np.mean(abs(loo))))
  f,e,w,l=min(fits,key=lambda u:u[0]);print('GP',metric,col,'valid',e,'loo',l,'noise',w[-1],sorted(zip(w[:-1],names),reverse=True)[:17],flush=True);np.savez('/tmp/gp12_%s_%s.npz'%(metric,col),w=w,names=names)
# Low-dimensional single or two-stage combinations; compare random and contiguous development folds.
sets={'base':['HL','HQ','HX','ZL','ZQ','ZX','HZd','HZx'], 'mixed':['HL','HQ','HX','ZL','HZd','HZx'],'native':['HL','HQ','HX','ZL','ZQ','ZX','RZ'],'latent':['HL','HQ','HX','ZL','ZQ','ZX','RQ','RX'],'simple':['HL','HQ','HX','ZL'],'c1':['HL','HQ','HX','CL','CQ','CX'],'c2':['HL','HQ','HX','DL','DQ','DX'],'allc':['HL','HQ','HX','CL','CQ','CX','DL','DQ','DX'],'c12':['CL','CQ','CX','DL','DQ','DX'],'hd3':['HL','HQ','HX','ZL','HZd','HZx','HD3'],'native2':['HL','HQ','HX','ZL','ZQ','ZX','RZ','RQ','RX']}
for g in ['g','j','p','q']:
 sets[g]=[g+'L',g+'Q',g+'X','ZL','ZQ','ZX',g+'cross'];sets[g+'H']=['HL','HQ','HX','ZL',g+'L',g+'Q',g+'X',g+'cross'];sets[g+'Hl']=['HL','HQ','HX','ZL',g+'L',g+'cross']
rows=[]
for k,kk in sets.items():
 inds=[keys.index(v) for v in kk];
 for w0 in [.015,.1,.5,1.,4.]:
  w=np.ones(len(inds));w[3:]=w0
  A=np.einsum('i,ijk->jk',w,K[inds])+K0
  for r in [1e-7,3e-6,3e-5,.0003,.003,.03]:
   cf=cho_factor(A[:n]+r*I); al=cho_solve(cf,Y[:n]); E=np.mean(abs(A[n:]@al-Y[n:]),axis=0); l=np.mean(abs(al/np.diag(cho_solve(cf,I))[:,None]),axis=0); rows.append((E[0],l[0],E[1:].mean(),l[1:].mean(),k,w0,r))
print('manual gate',sorted(rows)[:15]); print('manual loo',sorted(rows,key=lambda a:a[1])[:15]); print('manual traj',sorted(rows,key=lambda a:a[2])[:8]);

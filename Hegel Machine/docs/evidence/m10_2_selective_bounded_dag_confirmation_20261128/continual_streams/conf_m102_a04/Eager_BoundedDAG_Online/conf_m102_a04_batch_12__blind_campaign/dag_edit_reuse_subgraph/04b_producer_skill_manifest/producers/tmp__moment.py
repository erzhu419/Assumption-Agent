exec(open('/tmp/compose.py').read().split('features={};')[0])
from itertools import combinations_with_replacement as cr
from scipy.linalg import qr
import importlib.util
sp=importlib.util.spec_from_file_location('target','/holdout_task/_frozen_nodes/node_01.py'); t=importlib.util.module_from_spec(sp); sp.loader.exec_module(t)
print('Y singular',np.linalg.svd(y,compute_uv=False).round(5))
a=z; bb=bp(z); bi0=coords['Ic']; bc=coords['Bc']; P=prod(z); C=cos(z)
poly=lambda a,d:np.column_stack([np.prod(a[:,i],axis=1) for i in cr(range(a.shape[1]),d)])
Q=poly(a,2); mono=poly(a,4); rat=lambda a,k:a*a/(1+k*a*a)
blocks={'x':a,'B':bb,'I':bi0,'i':coords['I'],'b':bc,'P':P,'C':C,'Q':Q,'M':mono,'R':rat(a,.25),'RI':rat(bi0,.25),'RB':rat(bc,.25),'c':np.cos(q*a),'s':np.sin(q*a),'F':np.cos(2.15*a),'FB':np.cos(2.15*bb),'fi':np.cos(q*bi0),'fB':np.cos(q*bc),'t':np.tanh(a),'A':np.abs(a),'v':np.c_[np.cos(q*a),np.sin(q*a)],'E':np.c_[a*a,np.cos(q*a),np.cos(q*bi0),np.cos(q*bc)],'H':np.column_stack([np.cos(q*(a[:,i]+s*a[:,j])) for i,j in I.combinations(range(4),2) for s in (1,-1)]),'QC':np.einsum('ni,nj->nij',Q,C).reshape(n,-1),'PC':np.einsum('ni,nj->nij',P,C).reshape(n,-1)}
blocks['V']=np.c_[P,C]; blocks['J']=np.c_[P,C,Q]; blocks['O']=np.c_[a,P,C]; blocks['U']=np.c_[Q,blocks['E']]

def lin_score(w,ridge):
 out=[]
 for tr,te in pairs:
  W=(w-w[tr].mean(0))/np.maximum(w[tr].std(0),1e-7); mu=y[tr].mean(0); A=W[tr]; b=y[tr]-mu
  yp=W[te]@A.T@np.linalg.solve(A@A.T+ridge*np.eye(len(tr)),b)+mu
  e=abs(yp-y[te]); out.append((e[:,0].mean(),e[:,1:].mean()))
 return np.array(out)
res=[]; klist=list(blocks); basek=['V','Q','J','U','O','E','P','C',''];
for k in basek:
 for g in ['']+klist:
  if k==g:continue
  v=np.column_stack([blocks[i] for i in [k,g] if i]);
  configs=[('linear',v),('quad',np.c_[v,poly(v,2)])] if v.shape[1]<=21 else [('linear',v)]
  for label,V in configs:
   for r in [1e-6,.001,.03,.4]:
    R=lin_score(V,r); res.append((R.max(),R[0].max(),k,g,label,r,R))
  if v.shape[1]<=30:
   for fac in [.25,.5,1,2,4]:
    # Rescale the metric, keeping the two frozen kernel formulas unchanged.
    # score() normally normalizes, so apply scaling after normalization here.
    out=[]
    for tr,te in pairs:
     Z=fac*(v-v[tr].mean(0))/np.maximum(v[tr].std(0),.1); G=Z@Z.T/Z.shape[1]; d=np.diag(G); DD=np.maximum(d[:,None]+d-2*G,0); K=(1+.25*G)**2; L=K+np.exp(-.06*DD); eye=np.eye(len(tr)); ii=np.ix_(tr,tr); jj=np.ix_(te,tr); yp=(K[jj]@np.linalg.solve(K[ii]+3e-5*eye,y[tr])+L[jj]@np.linalg.solve(L[ii]+3e-4*eye,y[tr]))/2; e=abs(yp-y[te]); out.append((e[:,0].mean(),e[:,1:].mean()))
    R=np.asarray(out); res.append((R.max(),R[0].max(),k,g,'kernel',fac,R))
for mode in [0,1]:
 print('TOP',mode)
 for rr in sorted(res,key=lambda z:z[mode])[:16]:print(*rr[:6],rr[6].round(4).tolist())
# Single-coordinate power and harmonic coefficient diagnostics on the even subspace.
for name,w in [('quad',Q),('quad+cos',np.c_[Q,C]),('quad+quartic',np.c_[Q,mono]),('quad+cos+products',np.c_[Q,C,blocks['PC']])]:
 A=np.c_[np.ones(n),w]; c=np.linalg.lstsq(A,y,rcond=1e-9)[0]; err=A@c-y; print(name,'fit MAE',np.mean(abs(err[:,0])),np.mean(abs(err[:,1:])),'linear CV',lin_score(w, .001).round(4).tolist());
 if name=='quad+cos':print('coeffs',c[:,[0,-1]].round(4))

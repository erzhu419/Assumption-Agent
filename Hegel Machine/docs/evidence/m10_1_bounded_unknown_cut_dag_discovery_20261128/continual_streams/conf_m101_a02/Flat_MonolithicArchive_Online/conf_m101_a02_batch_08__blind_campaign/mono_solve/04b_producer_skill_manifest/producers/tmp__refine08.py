from campaign_probe import *
from itertools import combinations_with_replacement
np.set_printoptions(precision=4,suppress=True,linewidth=150)
z,b,c,d=(a[:n] for a in old._coords(s)); y=tr['y']; T=tr['traj']; P=z*np.roll(b,1,axis=1); a=P.sum(1)
u,v=s['panel_b']['u'],s['panel_b']['v']; print('panel B range',u.min(),u.max());
def phi(q,p): return q*(1-p[1]*np.exp(-p[0]*q*q))
for label,fun,p0 in [('base',lambda q,p:aff(phi(q,p)),[1,1.1]),('offset',lambda q,p:aff(q*(1-p[1]*np.exp(-p[0]*(q-p[2])**2))),[1,1.1,0]),('exponent',lambda q,p:aff(q*(1-p[1]*np.exp(-p[0]*abs(q)**p[2]))),[1,1.1,2]),('even',lambda q,p:np.c_[aff(phi(q,p)),q*q],[1,1.1]),('cross',lambda q,p:np.c_[aff(phi(q,p)),q*np.roll(q,1,axis=1)],[1,1.1]),('odd3',lambda q,p:np.c_[aff(phi(q,p)),q**3],[1,1.1]),('gauss/rat',lambda q,p:aff(q*(1-p[1]*np.exp(-p[0]*q*q))/(1+p[2]*q*q)),[1,1.1,0.1]),('gauss+sin',lambda q,p:aff(q-p[1]*q*np.exp(-p[0]*q*q)+p[2]*np.sin(p[3]*q)),[1,1.1,0,2])]:
 def fitb(ix):
  def err(p):
   f=fun(u[ix],p);return (f@fit(f,v[ix])-v[ix]).ravel()
  p=least_squares(err,p0,max_nfev=400).x; return p,fit(fun(u[ix],p),v[ix])
 p,w=fitb(np.arange(len(u))); err=v-fun(u,p)@w; loo=[]
 for k in range(len(u)):
  q,j=fitb(np.arange(len(u))!=k); loo.append((fun(u[k:k+1],q)@j-v[k]).ravel())
 B=fun(z,p)@w; J=(z*np.roll(B,1,axis=1)).sum(1); F=np.c_[aff(B),J]; print('B',label,p,np.std(err),np.std(loo),np.sort(np.linalg.norm(err,axis=1))[-4:]); print('readout',[(np.mean(abs(F[l:]@fit(F[:l],y[:l])-y[l:])),np.mean(abs(F[l:]@fit(F[:l],T[:l])-T[l:]))) for l in [52,56]])
base=np.c_[aff(b),a]; blocks={'z':z,'b':b,'c':c,'d':d,'x':tr['x'],'P':P}; names=[]; fs=[]
for key,q in blocks.items():
 for fn,f in [('id',q),('sq',q*q),('cube',q**3),('sin',np.sin(1.7*q)),('cos',np.cos(1.7*q)),('tanh',np.tanh(1.7*q)),('rat',q/(1+.2*q*q)),('exp',q*np.exp(-1.7*q*q))]:
  for j in range(4): names.append(f'{key}.{fn}.{j}'); fs.append(f[:,j]);
 for k in [1,2,3]:
  names.append(f'{key}.roll{k}'); fs.append((q*np.roll(q,k,axis=1)).sum(1))
for qa,qb in combinations_with_replacement(blocks,2):
 for roll in range(4):
  f=blocks[qa]*np.roll(blocks[qb],roll,axis=1)
  for w,g in [('sum',f.sum(1)),('sqsum',(f*f).sum(1))]:names.append(f'{qa}{qb}.{roll}.{w}');fs.append(g)
for lab,F in [('b+cyclic',base),('z+b+cyclic',np.c_[base,z]),('b+cyclic+c',np.c_[base,c]),('b+cyclic+d',np.c_[base,d])]:
 for l in [52,56]:
  arr=np.column_stack(fs); ix=[]; print('baseline',lab,l,np.mean(abs(F[l:]@fit(F[:l],y[:l])-y[l:])),np.std(F[:l]@fit(F[:l],y[:l])-y[:l]));
  for i in range(7):
   G=np.c_[F,arr[:,ix]]; r=y[:l]-G[:l]@fit(G[:l],y[:l]); R=arr[:l]-G[:l]@fit(G[:l],arr[:l]); norm=np.linalg.norm(R,axis=0); score=abs(R.T@r)/np.maximum(norm,1e-10); score[norm<.005]=0; score[ix]=0
   if i in [0,1,2,4,6]: print('step',i,np.std(r),np.mean(abs(G[l:]@fit(G[:l],y[:l])-y[l:])),[names[j] for j in ix],[names[j] for j in np.argsort(score)[-4:][::-1]])
   ix.append(np.argmax(score))

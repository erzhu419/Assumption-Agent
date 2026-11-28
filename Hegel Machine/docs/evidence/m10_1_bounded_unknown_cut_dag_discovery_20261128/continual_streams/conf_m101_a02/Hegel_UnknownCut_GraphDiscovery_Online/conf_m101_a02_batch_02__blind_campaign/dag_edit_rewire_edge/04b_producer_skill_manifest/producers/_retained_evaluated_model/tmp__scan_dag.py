import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe_dag import *
from sklearn.model_selection import KFold
from scipy.linalg import qr

def dictionary(x,z):
 W=np.column_stack((x,z)); names=[f'{a}{i}' for a in 'xz' for i in range(4)]; D={}
 for i,j in it.combinations_with_replacement(range(8),2): D[f'{names[i]}*{names[j]}']=W[:,i]*W[:,j]
 for i,j,k in it.combinations_with_replacement(range(8),3): D[f'{names[i]}*{names[j]}*{names[k]}']=W[:,i]*W[:,j]*W[:,k]
 for i,j in it.combinations_with_replacement(range(8),2):
  a,b=W[:,i],W[:,j]; n,h=names[i],names[j]
  for f in (.5,1,1.5,2,3,4,5,6):
   D[f'{n}*sin({f}*{h})']=a*np.sin(f*b); D[f'{h}*sin({f}*{n})']=b*np.sin(f*a)
   D[f'{n}/(1+{f}*{h}^2)']=a/(1+f*b*b); D[f'{h}/(1+{f}*{n}^2)']=b/(1+f*a*a)
 for i,j in it.combinations(range(8),2):
  for g in (-1,-.5,.5,1):
   a=W[:,i]+g*W[:,j]; n=f'({names[i]}+{g}*{names[j]})'
   for f in (.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6):
    D[f'sin({f}*{n})']=np.sin(f*a);D[f'cos({f}*{n})']=np.cos(f*a)
 for i in range(8):
  a=W[:,i];n=names[i]
  for f in (.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8):
   D[f'sin({f}*{n})']=np.sin(f*a);D[f'cos({f}*{n})']=np.cos(f*a);D[f'tanh({f}*{n})']=np.tanh(f*a)
 return list(D), np.column_stack(list(D.values()))

def path(A,D,Y,names,tr,te,steps=12):
 B=A.copy(); chosen=[]; rows=[]
 for itn in range(steps+1):
  BT=B[tr]; c=fit(BT,Y[tr]); r=Y[tr]-BT@c
  pred=B[te]@c; rows.append([np.mean(abs(r),axis=0),np.mean(abs(pred-Y[te]),axis=0),names[chosen[-1]] if chosen else 'affine'])
  Q=np.linalg.qr(BT,mode='reduced')[0]; DT=D[tr]; P=DT-Q@(Q.T@DT); norm=np.linalg.norm(P,axis=0); score=abs(P.T@r)/np.maximum(norm,1e-10); idx=np.argmax(score); chosen.append(idx); B=np.column_stack([B,D[:,idx]])
 return rows

print('official split:', next(l for l in open('/holdout_task/test_dag_candidate.py') if 'sp_idx = ' in l).strip())
for label,xx,zz,Y in [('labels',x,z,y)]+[(f'v1[{i}]',q,aff(q)@m,p['v1'][:,i]) for i in range(4)]:
 names,D=dictionary(xx,zz); n=len(xx); A=aff(xx); tr=np.arange(n-18 if n>30 else n); te=np.arange(len(tr),n) if n>30 else tr
 print(label, 'signed mixed residual', (Y-zz@np.array([-1,-1,1,1]))[:8] if n>30 else '');
 for limit in [120,len(names)]:
  rows=path(A,D[:,:limit],Y,names,tr,te,9); print('limit',limit,'path',[(round(a,4),round(b,4),c) for a,b,c in rows]);
  if n>30:
   v=[path(A,D[:,:limit],Y,names,tr,te,9) for tr,te in KFold(6,shuffle=True,random_state=500).split(A)];print('six fold mean / worst',np.round(np.mean([[r[1] for r in k] for k in v],axis=0),4),np.round(np.max([[r[1] for r in k] for k in v],axis=0),4))
# Use exact squares as the held-out reference for whether the second panel output needs cross terms.
for mat in (np.column_stack([aff(q),q*q]), np.column_stack([aff(q),q*q,(aff(q)@m)**2])):
 print('v2 squares loo',loo(mat,p['v2']))

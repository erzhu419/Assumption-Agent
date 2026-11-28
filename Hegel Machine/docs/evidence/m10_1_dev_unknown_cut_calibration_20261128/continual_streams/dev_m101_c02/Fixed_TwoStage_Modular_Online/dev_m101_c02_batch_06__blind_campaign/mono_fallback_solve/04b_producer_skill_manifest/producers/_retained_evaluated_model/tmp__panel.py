import numpy as np,pickle,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
E=sorted([e for e in it.product(range(5),repeat=4) if sum(e)<=4],key=lambda e:(sum(e),e))
def design(x,d): return np.column_stack([np.prod(x**e,axis=1) for e in E if sum(e)<=d])
print('telemetry',s['telemetry_vec'])
for p,k in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 x=s[p]['u']; y=s[p][k]; keep=np.arange(len(x))
 if p=='panel_a': keep=np.flatnonzero(np.linalg.norm(y-x,axis=1)<3)
 A=design(x[keep],4); y=y[keep]; print(p,k,len(y))
 for d,m in [(1,5),(2,15),(3,35),(4,70)]:
  if m>=len(y): continue
  M=A[:,:m]; C=np.linalg.lstsq(M,y,rcond=None)[0]; Q=np.linalg.qr(M)[0]; R=y-M@C
  print(d,'rms',np.sqrt(np.mean(R**2)),'loo',np.sqrt(np.mean((R/(1-(Q*Q).sum(1))[:,None])**2)))
  if d>=2:
   for j in range(4): print(j,[( ''.join(str(i) for i,n in enumerate(E[z]) for _ in range(n)) or '1',round(C[z,j],3)) for z in np.argsort(abs(C[:,j]))[-10:][::-1]])
 x=x[keep]
 for j in range(4):
  m=5; inds=list(range(m)); target=y[:,j]; residual=target-A[:,:m]@np.linalg.lstsq(A[:,:m],target,rcond=None)[0]; P=A-A[:,:m]@np.linalg.lstsq(A[:,:m],A,rcond=None)[0]
  path=[]
  for _ in range(5):
   score=abs(P.T@residual)/np.maximum(1e-8,np.linalg.norm(P,axis=0)); score[inds]=-1; i=np.argmax(score); inds.append(i); D=A[:,inds]; coef=np.linalg.lstsq(D,target,rcond=None)[0]; residual=target-D@coef; P=A-D@np.linalg.lstsq(D,A,rcond=None)[0]; path.append((E[i],round(coef[-1],3),round(np.std(residual),5)))
  print('sparse',j,path)

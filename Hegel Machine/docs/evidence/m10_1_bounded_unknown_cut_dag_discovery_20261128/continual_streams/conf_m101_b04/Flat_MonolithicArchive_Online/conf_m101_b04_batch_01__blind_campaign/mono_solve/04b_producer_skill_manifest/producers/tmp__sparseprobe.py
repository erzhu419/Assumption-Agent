from dagprobe import *
from scipy.linalg import qr
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; n=len(x); D=blocks(x,s); Q=poly(x,2); Y=np.c_[y,T[:,-3:].mean(1)]; ix=np.arange(n); splits=[(ix[:52],ix[52:])]+[(np.setdiff1d(ix,j),j) for j in np.array_split(np.random.default_rng(13).permutation(n),4)]; dd={}
def add(key,f):
 if np.std(f)>1e-8: dd[key]=f
for key in ('x','a','b','ca','da','cb','db'):
 z=D[key]; w=z/np.maximum(z.std(0),.1)
 dirs=[(str(i),z[:,i]) for i in range(4)]; dirs += [(f'{i}:{c}:{j}',z[:,i]+c*z[:,j]) for i,j in it.combinations(range(4),2) for c in [-2,-1,-.5,.5,1,2]]
 for a0,v in dirs:
  for k in [.6,1,1.6,2.2,3.2]:
   for label,f in [('sin',np.sin(k*v)),('cos',np.cos(k*v)),('tanh',np.tanh(k*v)),('rat',1/(1+(k*v)**2)),('odd',v/(1+(k*v)**2))]:
    add((key,a0,k,label),f)
    if len(a0)==1:
     for j in range(4):
      add((key,a0,k,label,j), f*z[:,j]); add((key,a0,k,label,'a',j),f*D['a'][:,j])
 for deg in [2,3,4]:
  for ids in it.combinations_with_replacement(range(4),deg): add((key,'poly',ids),np.prod(z[:,ids],axis=1))
 for j in range(4):
  for k in range(4):
   add((key,'crossa',j,k),z[:,j]*D['a'][:,k]);add((key,'crossb',j,k),z[:,j]*D['b'][:,k])
F=np.column_stack(list(dd.values())); names=list(dd); results={};
for i,j in splits:
 qi=qr(Q[i],mode='economic')[0]; G=F[i]-qi@(qi.T@F[i]); scale=np.linalg.norm(G,axis=0); scale=np.where(scale>1e-6,scale,1e20); G=G/scale; R=Y[i]-qi@(qi.T@Y[i]);
 for w in [0,.25,1,4,16]:
  for pool in [('x','a','b'),('a','b','ca','da','cb','db'),tuple(D)]:
   mask=np.array([k[0] in pool for k in names]); selected=[]; rr=R.copy()
   for k in range(1,13):
    d=G.T@rr; score=d[:,0]**2+w*d[:,1]**2; score[~mask]=-1; score[selected]=-1; selected.append(np.argmax(score)); A=G[:,selected]; rr=R-A@ls(A,R);
    if k in [1,2,3,4,5,6,8,10,12]:
     B=np.c_[Q,F[:,selected]]; b=ls(B[i],Y[i]); p=B[j]@b; results.setdefault((w,pool,k),[]).append((np.mean(abs(p-Y[j]),0),np.std(rr,0),[names[l] for l in selected]))
for kind in (0,1,2):
 print('ranking',kind)
 for key,v in sorted(results.items(),key=lambda it: [it[1][0][0][0],max(r[0][0] for r in it[1]),it[1][0][0][1]][kind])[:6]: print(key,np.array([r[0] for r in v]).round(4),v[0][1].round(4),v[0][2])

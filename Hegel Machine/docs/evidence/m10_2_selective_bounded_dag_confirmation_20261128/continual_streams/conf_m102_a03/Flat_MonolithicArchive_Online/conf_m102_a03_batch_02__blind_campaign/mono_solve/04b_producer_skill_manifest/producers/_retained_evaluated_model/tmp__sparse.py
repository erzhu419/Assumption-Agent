exec(open('/tmp/discover.py').read().split('A=old._calibration')[0]);from scipy.optimize import least_squares
from itertools import combinations
A=old._calibration(s['panel_a']);z=old._decode(x,A);pb=s['panel_b'];B=fit(poly(pb['u'],1),pb['v']);w=poly(z,1)@B;pc=s['panel_c'];U=pc['u'];V1=pc['v1'];V2=pc['v2'];print('telemetry',s['telemetry_vec'], 'times',s['eval_times']);D=lambda z:poly(z,1)
fa=lambda f,p:np.c_[D(f),np.arctan(p[0]*f),np.sin(p[1]*f)]
fb=lambda f,p:np.c_[np.ones(len(f)),f+p[1]*np.cos(p[0]*f)]
def infer(F,V,p):
 def r(p):
  P=F(U,p);return (P@fit(P,V)-V).ravel()
 p=least_squares(r,p).x;C=fit(F(U,p),V);return lambda f:F(f,p)@C
f1=infer(fa,V1,[4.4,2.58]);f2=infer(fb,V2,[1.28,-.65]);a=f1(z);b=f2(z)
rng=np.random.default_rng(29);perm=rng.permutation(72); folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),perm[i:i+18]),perm[i:i+18]) for i in range(0,72,18)]
for i,j in folds[:1]:
 for n,F in [('a',a),('b',b),('a+b',a+b),('a-b',a-b),('a,b',np.c_[a,b]),('a,b,w',np.c_[a,b,w]),('a,b,ab',np.c_[a,b,a*b])]:
  C=fit(D(F)[i],y[i]);print(n,C.round(3),np.mean(abs(D(F)[j]@C-y[j])))
lib={}
def add(n,F):
 for k,c in enumerate(F.T):lib[f'{n}{k}']=c
for n,F in [('z',z),('w',w),('a',a),('b',b),('s',a+b),('d',a-b)]:
 for nn,G in [('',F),('^2',F**2),('^3',F**3),('tanh',np.tanh(F)),('sin',np.sin(1.25*F)),('cos',np.cos(1.25*F)),('rat',F/(1+.22*F**2))]:add(n+nn,G)
 for k,l in combinations(range(4),2):lib[f'{n}{k}*{n}{l}']=F[:,k]*F[:,l]
for k in range(4):
 for l in range(4):
  lib[f'a{k}*b{l}']=a[:,k]*b[:,l];lib[f'a{k}/b{l}']=a[:,k]/(1+.22*b[:,l]**2);lib[f'b{l}/a{k}']=b[:,l]/(1+.22*a[:,k]**2)
for n,F in [('aa',a),('bb',b)]:
 for k,l in combinations(range(4),2):
  lib[f'{n}{k}/({l})']=F[:,k]/(1+.22*F[:,l]**2);lib[f'{n}{l}/({k})']=F[:,l]/(1+.22*F[:,k]**2)
names=np.array(list(lib));M=np.column_stack(list(lib.values()));
for base in [np.ones((72,1)),D(z),np.c_[D(z),a,b],np.c_[D(z),a,b,b*b],np.c_[D(z),a,b,a*b,a*a,b*b]]:
 print('base',base.shape[1]);by=[]
 for i,j in folds:
  Q=np.linalg.qr(base[i],mode='reduced')[0];P=M[i]-Q@(Q.T@M[i]);P/=np.linalg.norm(P,axis=0).clip(1e-10);sel=[];F=base
  vals=[np.mean(abs(F[j]@fit(F[i],y[i])-y[j]))];train=[]
  for k in range(12):
   r=y[i]-F[i]@fit(F[i],y[i]);q=abs(P.T@r);q[sel]=-1;l=np.argmax(q);sel.append(l);F=np.c_[base,M[:,sel]]; vals.append(np.mean(abs(F[j]@fit(F[i],y[i])-y[j])));train.append(np.mean(abs(r)))
  by.append(vals)
  if len(by)==1:print(list(zip(names[sel],np.round(train,4),np.round(vals[1:],4))))
 print('worst',np.max(by,axis=0).round(5),'mean',np.mean(by,axis=0).round(5))

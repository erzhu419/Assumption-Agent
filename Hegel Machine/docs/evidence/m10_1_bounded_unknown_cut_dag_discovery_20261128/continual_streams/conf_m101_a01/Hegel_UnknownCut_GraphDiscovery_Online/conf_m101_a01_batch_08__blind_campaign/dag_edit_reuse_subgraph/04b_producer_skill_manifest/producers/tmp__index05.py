exec(open('/tmp/probe05.py').read().split('for lbl,A in')[0])
from scipy.optimize import least_squares
n=len(q); Q=poly(q); H=Q[:,5:]; splits=[('official',np.arange(n-20),np.arange(n-20,n))]
for k in range(24):
 r=np.random.default_rng(k).permutation(n);splits.append((str(k),r[:-20],r[-20:]))
for k in range(4):
 r=q[:,k].argsort();splits.extend([(f'{k}low',r[20:],r[:20]),(f'{k}high',r[:-20],r[-20:])])
def fit(i,j,kind,structure,Y=y):
 Z={'linear':Q[:,1:5],'quadratic':Q[:,1:],'pairs':np.c_[q,H[:,[1,2,3,5,6,8]]]}[structure]; core=Q[:,1:5] if structure=='quadratic' else Q[:,1:]; W=np.c_[np.ones(n),core]; blocks=np.linalg.qr(W[i])[0]; yi=Y[i]; yi=yi-blocks@(blocks.T@yi)
 def calc(p,k):
  t=p[0]+Z[k]@p[1:];return {'sin':np.sin,'tanh':np.tanh,'cos':np.cos,'soft':lambda x:x/np.sqrt(1+x*x)}[kind](t)
 def loss(p):
  v=calc(p,i);v=v-blocks@(blocks.T@v); coef=(v@yi)/(v@v+1e-9);return np.r_[coef*v-yi,.002*p]
 w=np.linalg.lstsq(Q[i],Y[i],rcond=None)[0]; d=Z.shape[1]; rng=np.random.default_rng(15); guesses=[np.r_[0.,w[1:1+d]] if structure!='pairs' else np.r_[0.,w[1:5],w[[6,7,8,10,11,13]]]]
 guesses.extend(rng.normal(0,.8,(7,d+1))); cand=min((least_squares(loss,p,max_nfev=105) for p in guesses), key=lambda a:a.cost); p=cand.x
 A=np.c_[W[i],calc(p,i)];c=np.linalg.lstsq(A,Y[i],rcond=None)[0];return np.c_[W[j],calc(p,j)]@c,p,c
for struct in ['linear','pairs','quadratic']:
 for kind in ['sin','tanh','cos','soft']:
  es=[]
  for name,i,j in splits:
   yh,p,c=fit(i,j,kind,struct); es.append(np.mean(abs(yh-y[j])))
  _,p,c=fit(np.arange(n),np.arange(n),kind,struct); print(struct,kind,np.round([es[0],np.mean(es[1:25]),max(es[1:25]),np.mean(es[25:]),max(es[25:])],5),'p',np.round(p,2),'amp',round(c[-1],3),flush=True)
print('trajectory singular values',np.linalg.svd(T,compute_uv=False))

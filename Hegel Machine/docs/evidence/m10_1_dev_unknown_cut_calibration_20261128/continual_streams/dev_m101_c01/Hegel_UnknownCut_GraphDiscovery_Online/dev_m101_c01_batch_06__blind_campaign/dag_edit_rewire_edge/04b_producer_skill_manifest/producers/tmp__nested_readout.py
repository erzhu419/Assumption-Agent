exec(open('/tmp/compact.py').read().split('# Composed')[0].split('recs=[]')[0])
# Calibrated panel maps are independent of the study labels. Select the
# scalar kernel combination by inner LOO within each outer training block.
G = build(B(x), cal['v1'][0]); H = build(B(x), cal['v2'][0]);
U = np.column_stack([x,G,H]); ug=U-U.mean(0); ug/=np.maximum(U.std(0),1e-8)
def normalized(K):
 d=K.diagonal(); return K / np.mean(d)
Q=normalized(quad(x)@quad(x).T); M=banks['base'][2,:n,:n]; R=banks['base'][1,:n,:n]; W=banks['wide'][4,:n,:n];
P=sum(normalized(quad(Z)@quad(Z).T) for Z in (G,H))/2
L=normalized(ug@ug.T); J=normalized(L**2)
C=np.exp(-cdist(ug,ug,'sqeuclidean')/80)
models={
 'smooth': [8*Q+K for K in (M,R,W,.5*(M+W))],
 'panel': [8*Q+w*P+K for w in (.5,4,16) for K in (M,R,W,C)],
 'cross': [8*Q+w*(P+J)+K for w in (.5,4,16) for K in (M,R,W,C)]}

def baseline(tr,te):
 f=quad(x); fac=cho_factor(M[np.ix_(tr,tr)]+.001*np.eye(len(tr))); X=f[tr]; Z=cho_solve(fac,X); V=cho_solve(fac,Y[tr]); w=np.linalg.solve(X.T@Z,X.T@V); return f[te]@w+M[np.ix_(te,tr)]@(V-Z@w)

def nested(tr,te,family):
 best=float('inf'); result=None
 for K in family:
  a=K[np.ix_(tr,tr)]; b=K[np.ix_(te,tr)]; ev,vec=np.linalg.eigh(a); cy=vec.T@Y[tr];
  for reg in (.0001,.0004,.0015,.005,.015,.045):
   den=1/(ev+reg); weights=vec@(den[:,None]*cy); score=np.mean(abs(weights[:,0]/((vec**2)@den))); 
   if score<best: best=score; result=b@weights
 return result

def summary(name, fn):
 v=[]
 for tr,te in splits:
  pred=fn(tr,te); mode=np.linalg.svd(Y[tr,1:],full_matrices=False)[2][0]; rank=(pred[:,1:]@mode)[:,None]*mode; err=abs(pred-Y[te]); v.append([err[:,0].mean(),err[:,1:].mean(),abs(rank-Y[te,1:]).mean()]);
 v=np.asarray(v); print(name,'official/random mean/random max/edge',*[q.round(6).tolist() for q in (v[0],v[1:-1].mean(0),v[1:-1].max(0),v[-1])]); return v
summary('baseline',baseline)
for name, Ks in models.items(): summary(name,lambda tr,te:nested(tr,te,Ks))
summary('nested ensemble',lambda tr,te:.5*baseline(tr,te)+.25*nested(tr,te,models['panel'])+.25*nested(tr,te,models['cross']))

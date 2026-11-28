import pickle,numpy as np
R=pickle.load(open('/tmp/node03_nested.pkl','rb'))
L=pickle.load(open('/tmp/node03_loo.pkl','rb'))
A=pickle.load(open('/tmp/node03_folds.pkl','rb'))
def args(row):
 p,b,vt,y,tr=row;s=b[:,0];d=p[:,0]-s;D=(p[:,1:]-b[:,1:])@vt
 return s,d,D,y
A=[(a['poly'],b['smooth'],vt,y,t) for a,b,y,t,vt in A]
rows=[r for _,r in R]+[r for _,r in L[:4]]+A[1:]
def evaluate(name,fun):
 z=[]
 for r in rows:
  s,d,D,y=args(r);z.append(np.mean(abs(s+fun(s,d,D)-y)))
 return (name,np.array(z))
models=[evaluate('old',lambda s,d,D:.65*d),evaluate('smooth',lambda s,d,D:0)]
for w in (.25,.5,.75,1.):
 for off in (0,-.01,-.02,-.03):
  models.append(evaluate(('bias',w,off),lambda s,d,D:w*d+off))
 for q in (.02,.04,.08):
  for off in (0,-.01,-.02):
   for gate in ('saturate','clip','negative','posgate','concordance','spread'):
    def f(s,d,D):
     delta={'saturate':q*np.tanh(d/q),'clip':np.clip(d,-q,q),'negative':np.minimum(d,0),'posgate':np.minimum(d,0)+q*np.tanh(np.maximum(d,0)/q),'concordance':d/(1+(D[:,-1]/q)**2),'spread':d/(1+(d/q)**2)}[gate]
     return w*delta+off
    models.append(evaluate((gate,w,q,off),f))
for w in (.25,.5,.75):
 for cap in (.5,.75,1.0):
  for c in (-.04,-.02,.02,.04):
   models.append(evaluate(('magnitude',w,cap,c),lambda s,d,D:w*d+c*np.clip(abs(s)/cap,0,1)))
base=models[0][1]; print('baseline',np.round(base,5)); good=[(k,z) for k,z in models if z[4]<=.038]; print('passing',len(good));
for pool in (good,models):
 for k,z in sorted(pool,key=lambda k:np.mean(np.delete(k[1],4))+.2*max(np.delete(k[1],4)))[:14]: print(k,np.round(z,5),round(np.mean(np.delete(z,4)),6))
print('official best')
for k,z in sorted(models,key=lambda k:k[1][4])[:10]: print(k,np.round(z,5))

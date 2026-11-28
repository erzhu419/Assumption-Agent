import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z,b,c,d=m._features(s,tr['x']); y=tr['y']; Y=tr['traj']; st=np.column_stack; o=np.ones(len(z)); roll=z*np.roll(c,1,axis=1); cd=c*d
base=st((o,c,d,roll.sum(1),cd.sum(1))); broad=st((o,c,d,m._outer(z,c),cd.sum(1),(z*d).sum(1),m._outer(c,d)))
order=[np.arange(72),np.random.default_rng(19).permutation(72),np.random.default_rng(47).permutation(72),np.argsort(np.max(abs(z),axis=1))]
def score(D,V):
    a=[]
    for ix in order:
        i,j=ix[:52],ix[52:]; e=abs(D[j]@m._fit(D[i],V[i])-V[j]); a.append([e.mean(),e.mean(0).max()]);
    return np.array(a).round(6).tolist()
print('compact scalar',score(base,y)); print('broad state',score(broad,Y))
for key,D in {'compact':base,'roll-components':st((o,c,d,roll,cd.sum(1))),'compact-cd':st((base,cd)),'compact-fullcd':st((base,m._outer(c,d))),'compact-cd-zd':st((base,cd,(z*d).sum(1))),'compact-cd-fullzd':st((base,cd,m._outer(z,d))),'compact-cd-fullzc':st((base,cd,m._outer(z,c))),'compact-cd-z':st((base,cd,z))}.items():
  print(key,score(D,Y))
print('compact scalar coef',m._fit(base[:52],y[:52]).round(4))
D=st((o,c,d,roll.sum(1),cd.sum(1),m._outer(c,d),m._outer(z,c)))
for col in (0,3,7):
 print('state coeff',col,m._fit(D,Y[:,col]).round(3))

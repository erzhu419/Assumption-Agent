exec(open('/tmp/probe.py').read().split('ops=[]')[0])
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from scipy.linalg import solve
from scipy.optimize import least_squares
p=s['panel_a']; M=np.array([np.polyfit(p['u'][:,i],p['v'][:,i],1) for i in range(4)])
for it in range(5):
 r=np.linalg.norm(p['u']*M[:,0]+M[:,1]-p['v'],axis=1); good=np.argsort(r)[:40]; M=np.array([np.polyfit(p['u'][good,i],p['v'][good,i],1) for i in range(4)])
def panel(u,v,d):
 f=PolynomialFeatures(d); a=f.fit_transform(u); w=lstsq(a,v)[0]; return lambda x:f.transform(x)@w
p=s['panel_b']; b=panel(p['u'],p['v'],1);p=s['panel_c']; c1=panel(p['u'],p['v1'],1); c2=panel(p['u'],p['v2'],2)
a=x*M[:,0]+M[:,1]; inv=(x-M[:,1])/M[:,0]
P2=lambda z:PolynomialFeatures(2,include_bias=False).fit_transform(z)
P3=lambda z:PolynomialFeatures(3,include_bias=False).fit_transform(z)
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=217).split(x))
def calc(K,reg):
 out=[]
 for tr,te in folds:
  mu=Y[tr].mean(0); pred=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+reg*np.eye(len(tr)),Y[tr]-mu,assume_a='pos')+mu
  err=np.abs(pred-Y[te]);out.append([err[:,0].mean(),err[:,1:].mean(),err[:,-1].mean()])
 return np.asarray(out)
lin=lambda z:z@z.T
base=np.ones((len(x),len(x)))+lin(x)+.30*lin(P2(x)[:,4:]); cases=[]
for name,z in [('x',x),('a',a),('inv',inv),('b',b(a)),('b0',b(x)),('c1',c1(a)),('c2',c2(a)),('c2b',c2(b(a)))]:
 q=z-z.mean(0); d=cdist(q,q,'sqeuclidean'); prod=q@q.T
 nonlinear=[('purecube',lin(q**3)),('cubics',lin(P3(z)[:,14:])),('gauss',np.exp(-d/2)),('gausslong',np.exp(-d/6)),('tanh',lin(np.tanh(z))),('sin',lin(np.sin(z))),('sq',lin(z*z)),('poly',lin(P2(z))),('mix',lin(P2(np.c_[x,z])))]
 for nm,nl in nonlinear:
  for co in [.03,.3,3.]:
   for reg in [.0001,.001,.01,.05]:
    e=calc(base+co*nl,reg);cases.append((e[1:,0].mean(),e[0,0],e[:,1].max(),e[:,2].max(),name,nm,co,reg))
for ix in [0,1,2]:
 print('rank',ix)
 for row in sorted(cases,key=lambda z:z[ix])[:12]: print(row)

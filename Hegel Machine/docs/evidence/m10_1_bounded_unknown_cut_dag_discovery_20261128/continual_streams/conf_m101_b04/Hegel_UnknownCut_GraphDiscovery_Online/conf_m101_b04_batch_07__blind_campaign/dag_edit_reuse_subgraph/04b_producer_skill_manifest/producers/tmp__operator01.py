exec(open('/tmp/probe_node01.py').read().split("print('ranges'")[0])
from scipy.optimize import least_squares

def mat(a,f,bias=True):
 return np.column_stack(([np.ones(len(a))] if bias else [])+[a,f(a)])
def rational(a,p): return a**3/(1+p*a*a)
def panelopt(name,fn,p0):
 def error(p):
  A=mat(u,lambda a:fn(a,p)); return (A@np.linalg.lstsq(A,v,rcond=None)[0]-v).ravel()
 q=least_squares(error,p0,bounds=(.015,10),ftol=1e-12,gtol=1e-12,xtol=1e-12)
 A=mat(u,lambda a:fn(a,q.x)); c=np.linalg.lstsq(A,v,rcond=None)[0]; e=v-A@c; h=(np.linalg.qr(A)[0]**2).sum(1); print(name,q.x,'train',np.mean(abs(e)),'fixed-basis LOO',np.mean(abs(e/(1-h[:,None])))); return c,q.x
c,p=panelopt('rational shared',rational,[1.]); c4,p4=panelopt('rational per input',rational,[1.]*4)
for name,fn in [('tanh',lambda a,p:np.tanh(p*a)),('sin',lambda a,p:np.sin(p*a)),('power',lambda a,p:a**3/(1+p[0]*a*a)**p[1]),('arctan',lambda a,p:np.arctan(p*a))]: panelopt(name,fn,[1.,1.] if name=='power' else [1.])
print('Panel B shared rational coefficients',c,sep='\n'); print('Panel B per-input rational coefficients',c4,sep='\n')
b=mat(x,lambda a:rational(a,p))@c; b4=mat(x,lambda a:rational(a,p4))@c4
_,_,vt=np.linalg.svd(z,full_matrices=False); target=np.c_[y,z@vt[0]]
def score(name,A):
 pred=np.empty_like(target)
 for te in np.array_split(np.arange(len(x)),4):
  tr=np.setdiff1d(np.arange(len(x)),te); pred[te]=A[te]@np.linalg.lstsq(A[tr],target[tr],rcond=1e-8)[0]
 tr=np.arange(52); pub=A[52:]@np.linalg.lstsq(A[tr],target[tr],rcond=1e-8)[0]; e=np.abs(target-pred)
 print(name,A.shape[1],'mean',e.mean(0),'worst',np.max([f.mean(0) for f in np.array_split(e,4)],axis=0),'pub',abs(pub-target[52:]).mean(0),'in',fit(A,target)[1])
ones=np.ones(len(x)); forms={'x':x,'x3':x**3,'rational':rational(x,p),'b':b,'b4':b4,'b2':b*b,'b3':b**3,'x2':x*x,'xb':x*b,'qX':poly(x,2)[:,5:],'qB':poly(b,2)[:,5:],'qB4':poly(b4,2)[:,5:],'tanhB':np.tanh(b),'tanhX':np.tanh(x),'rB':b/(1+b*b),'rX':x/(1+x*x),'cosB':np.cos(b),'sinB':np.sin(b),'logB':np.log1p(b*b)}
for base in ('x','b','x x3','x rational','x b','x b4','x x3 b','x x3 b4'):
 for ext in ('',' x2',' qX',' b2',' qB',' qB4',' xb',' b3',' tanhB',' rB',' sinB',' b2 xb',' x2 xb',' b2 rB',' cosB',' logB'):
  key=base+ext; score(key,np.column_stack([ones]+[forms[n] for n in key.split()]))

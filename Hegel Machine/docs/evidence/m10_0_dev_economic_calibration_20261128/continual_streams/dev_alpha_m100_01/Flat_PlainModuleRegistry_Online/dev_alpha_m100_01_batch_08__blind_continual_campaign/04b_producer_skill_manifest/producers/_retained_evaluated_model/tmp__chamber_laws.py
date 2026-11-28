from chamber_data import *
from scipy.optimize import least_squares
F=tau[1]*((1-.19)-.127/tau[0]); print('linear rate-neutral slow slope',F)
print('sorted score, fast residual after full withdrawal, slow shift, drift, rate shift after withdrawal, fitted slow score')
z=h[:,None]-g
print(np.round(np.c_[s,dc[:,0]-th[0]*z[:,0],dc[:,1],s+dc[:,2],R[:,-1]-th[0]*z[:,0]/tau[0],dc[:,1]-F*s][np.argsort(s)],5))
base=c.copy(); base[:,0]=th[0]*h
Y=y-base@B.T
A=B.T
l=di['lab_id']
scale=np.r_[np.full(len(t),1.),1.]
# Direct observation-level fit: fast score, slow transfer, drift score.
def fit(kind, mask):
 def coeff(v):
  a,b,d,e=v[:4]; u=v[4] if len(v)>4 else 0.; w=v[5] if len(v)>5 else 0.
  slow={'lin':lambda: e+b*s,'quad':lambda:e+b*s+u*s*s,'cubic':lambda:e+b*s+u*s*s+w*s**3,'rat':lambda:e+b*s/(1+u*s*s),'sat':lambda:e+b*np.tanh(u*s)/u if u else e+b*s,'tanh0':lambda:e+b*np.tanh(u*s),'abs':lambda:e+b*s+u*abs(s),'history':lambda:e+b*s+u*z[:,0]+w*z[:,1],'biasedrat':lambda:e+b*(s+w)/(1+u*(s+w)**2),'ratfull':lambda:(e+b*s)/(1+u*s*s)}[kind]()
  return np.c_[a*s,slow,d*s] @ A
 k=dict(lin=4,quad=5,cubic=6,rat=5,sat=5,tanh0=5,abs=5,history=6,biasedrat=6,ratfull=5)[kind]; init=[.13,2.6,-.81,.33,.1,0.][:k]
 lo=[-np.inf]*k; hi=[np.inf]*k
 if kind in ('rat','sat','tanh0','biasedrat','ratfull'): lo[4]=0
 if kind=='biasedrat': lo[5],hi[5]=-2,2
 opt=least_squares(lambda v:((coeff(v)-Y)/scale)[mask].ravel(),init,bounds=(lo,hi),max_nfev=400)
 return opt.x,Y-coeff(opt.x)
for k in ('lin','quad','cubic','rat','sat','tanh0','abs','history','biasedrat','ratfull'):
 v,e=fit(k,np.ones(len(s),bool)); es=np.vstack([fit(k,l!=i)[1][l==i] for i in np.unique(l)])
 def stats(q):return np.round([np.sqrt(np.mean(q[:,:-1]**2)),np.sqrt(np.mean(q[:,-1]**2)),np.max(abs(q[:,:-1])),np.max(abs(q[:,-1]))],5)
 print(k,v.round(5),'in',stats(e),'LOLO',stats(es))

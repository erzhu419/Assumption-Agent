exec(open('/tmp/probe.py').read().split('res=[]')[0])
from scipy.special import erf
u=c['u']; v=c['v1']; print('C linear coefficients and R2',lin(one(u),v),1-((one(u)@lin(one(u),v)-v)**2).sum(0)/((v-v.mean(0))**2).sum(0)); print('v1',v.round(3));
coords={'u':u,'b':u@B[1:],'B':u@np.linalg.inv(B[1:]),'bf':one(u)@B,'Bi':(u-B[0])@np.linalg.inv(B[1:])}
funcs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'soft':lambda t:t/(1+abs(t)),'odd':lambda t:t/(1+t*t),'even':lambda t:1/(1+t*t),'erf':erf,'root':lambda t:t/np.sqrt(1+t*t),'square':np.square,'gauss':lambda t:np.exp(-t*t)}
out=[]
for tag,h in coords.items():
 for name,g in funcs.items():
  for extended in (False,True):
   f0=one(h) if extended else np.ones((len(h),1))
   def ff(q): return np.c_[f0,g(q[0]*h+q[1])]
   def rr(q):
    F=ff(q); return (F@lin(F,v)-v).ravel()
   opt=min((least_squares(rr,[k,p],bounds=([.001,-4],[20,4]),max_nfev=140) for k in (.45,1.25,2.5,4.5,7) for p in (0,1)),key=lambda r:r.cost)
   out.append((np.sqrt(2*opt.cost/v.size),tag,name,extended,opt.x))
print('fixed coordinates',*sorted(out,key=lambda a:a[0])[:20],sep='\n')
for j,y in enumerate(v.T):
 scores=[]; candidates=[]
 for name,g in funcs.items():
  if name=='square': continue
  for extended in (False,True):
   base=one(u) if extended else np.ones((len(u),1))
   def rr(p):
    F=np.c_[base,g(u@p[:4]+p[4])]; return F@lin(F,y)-y
   starts=[]
   for H in [np.eye(4), B[1:],np.linalg.inv(B[1:])]:
    for a in H.T:
     starts.extend([np.r_[a*k,p] for k in (1.25,3,5) for p in (0.,1.)]);
   rng=np.random.default_rng(29); starts.extend(rng.normal(size=(20,5))*2)
   opt=min((least_squares(rr,p,max_nfev=200) for p in starts),key=lambda r:r.cost)
   scores.append((np.sqrt(2*opt.cost/len(y)),name,extended,opt.x.round(4)))
 print('single-index',j,*sorted(scores,key=lambda a:a[0])[:6],sep='\n')

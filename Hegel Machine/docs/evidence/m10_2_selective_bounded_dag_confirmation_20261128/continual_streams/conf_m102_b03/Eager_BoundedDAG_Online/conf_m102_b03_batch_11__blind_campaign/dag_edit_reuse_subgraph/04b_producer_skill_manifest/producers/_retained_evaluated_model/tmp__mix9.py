exec(open('/tmp/panel_search.py').read().split('C={')[0])
from itertools import combinations
from scipy.special import expit
from scipy.spatial.distance import cdist
np.set_printoptions(precision=5,suppress=True)
print('align',a,b,'sv',np.linalg.svd(t,compute_uv=False),'n',len(y),'times',s['eval_times']);print('summary matches',np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()]))
Y=np.c_[y,t]; i=np.arange(52);j=np.arange(52,72); print('ranges',z.min(0),z.max(0))
R=[]
for ws,w in [('z',z),('x',x),('b',z@B+h),('r',(z-h)@P)]:
 c,d=(f(w) for f in funcs); cr=(c[:,:,None]*d[:,None,:]).reshape(len(w),-1); base=ones(np.c_[c,d,(c*d).sum(1)]); trend=ones(np.c_[c,d,cr]);
 def eval(F):
  coef=ls(F[i],Y[i]); E=abs(F@coef-Y); return E[j,0].mean(), E[j,1:].mean(), E[i,0].mean(), E[i,1:].mean()
 for k in range(4):
  if ws=='z':
   E=Y-trend@ls(trend,Y); ii=np.argsort(w[:,k]);print('bins',k,[np.round(E[b][:,[0,-1]].mean(0),3) for b in np.array_split(ii,8)])
 projections=[(str(k),w[:,k]) for k in range(4)];
 for k,m in combinations(range(4),2):
  for sign in [-1,1]:projections.append((f'{k}{sign:+d}{m}',w[:,k]+sign*w[:,m])); projections.append((f'{k}*{m}',w[:,k]*w[:,m]));
 projections += [('dot',(c*d).sum(1)), ('sum',w.sum(1)), ('norm',(w*w).sum(1)),('cnorm',(c*c).sum(1)),('dnorm',(d*d).sum(1))];
 for pn,p in projections:
  for shift in [0.,np.median(p),np.quantile(p,.3),np.quantile(p,.7)]:
   h=(p-shift)/max(p.std(),.0001)
   for gn,g in [('hard',h>0),('logit',expit(2.15*h)),('fast',expit(8*h)),('rat',1/(1+2.15*h*h)),('soft',h/(1+2.15*abs(h)))]:
    for dn,D in [('only',np.ones((len(w),1))),('z',ones(w)),('c',ones(c)),('d',ones(d)),('cd',ones(np.c_[c,d]))]:
     for bn,b in [('base',base),('trend',trend)]:R.append((*eval(np.c_[b,g[:,None]*D]),ws,pn,round(shift,3),gn,dn,bn))
print('y',sorted(R)[:10]);print('t',sorted(R,key=lambda v:v[1])[:10]);print('train y',sorted(R,key=lambda v:v[2])[:5]);print('train t',sorted(R,key=lambda v:v[3])[:5]);

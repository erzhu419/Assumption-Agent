from hegel_probe import *
from itertools import combinations
from scipy.optimize import root
lin=a.lin; one=a.one
u,v=p['panel_a'].values(); F=one(u); keep=np.ones(len(u),bool)
for _ in range(4):
 r=np.linalg.norm(v-F@lin(F[keep],v[keep]),axis=1); keep=r<2
u,v=u[keep],v[keep]; nxt=np.roll(u,-1,axis=1)
pairs=lambda w:np.column_stack([w[:,i]*w[:,j] for i,j in combinations(range(4),2)])
def AF(w,mode):
 r=np.roll(w,-1,axis=1)
 return {'cyc':np.c_[one(w),w*r,w**3], 'cycabs':np.c_[one(w),w*r,w*abs(w)],'poly':np.c_[one(w),w*w,pairs(w),w**3],'polyabs':np.c_[one(w),w*w,pairs(w),w*abs(w)]}[mode]
for mode in ['cyc','cycabs','poly','polyabs']:
 G=AF(u,mode); c=lin(G,v); print('A',mode,np.sqrt(np.mean((G@c-v)**2)),np.round(c,4))
for j in range(4):
 w=u[:,j]; z=nxt[:,j]; b=np.c_[np.ones(len(w)),w,z,w*z]; y=v[:,j];
 opts={'cube':lambda q:np.c_[w**3,z**3], 'oddabs':lambda q:np.c_[w*abs(w),z*abs(z)], 'cubic_mix':lambda q:np.c_[w**3,w*w*z,w*z*z,z**3], 'rational':lambda q:np.c_[w**3/(1+q[0]*w*w),z**3/(1+q[0]*z*z)], 'sine':lambda q:np.c_[np.sin(q[0]*w),np.sin(q[0]*z)], 'ratpair':lambda q:np.c_[w*z/(1+q[0]*w*w),w*z/(1+q[0]*z*z),w**3,z**3]}
 for k,fun in opts.items():
  def d(q):return np.c_[b,fun(q)]
  def r(q):
   g=d(q); return g@lin(g,y)-y
  opt=least_squares(r,[.4],bounds=(.0001,6),gtol=1e-12); print('Asparse',j,k,opt.x,np.std(opt.fun),np.round(lin(d(opt.x),y),4))
cu=p['panel_c']['u']; cv=p['panel_c']['v1']; cv2=p['panel_c']['v2']
cm={'softlin':lambda w,q:w/(1+q[0]*abs(w))+q[1]*w, 'tanhlin':lambda w,q:np.tanh(q[0]*w)+q[1]*w, 'softfree':lambda w,q:np.c_[w,w/(1+q[0]*abs(w))], 'softcubic':lambda w,q:np.c_[w,w/(1+q[0]*abs(w)),w**3], 'softtanh':lambda w,q:np.c_[w,w/(1+q[0]*abs(w)),np.tanh(q[1]*w)], 'tanhabs':lambda w,q:np.c_[w,np.tanh(q[0]*w),w*abs(w)], 'softsat':lambda w,q:np.c_[w,w/(1+q[0]*abs(w)+q[1]*w*w)], 'oddabsrat':lambda w,q:np.c_[w,w*abs(w)/(1+q[0]*abs(w)+q[1]*w*w)], 'softshift':lambda w,q:np.c_[w,(w+q[1])/(1+q[0]*abs(w+q[1]))], 'tanhshift':lambda w,q:np.c_[w,np.tanh(q[0]*(w+q[1]))], 'atan':lambda w,q:np.c_[w,np.arctan(q[0]*w)], 'softpower':lambda w,q:np.c_[w,w/(1+q[0]*abs(w)**q[1])], 'softper':lambda w,q:np.c_[w,w/(1+np.array(q)*abs(w))], 'tanhper':lambda w,q:np.c_[w,np.tanh(np.array(q)*w)]}
C={}
for k,fun in cm.items():
 def d(w,q):return one(fun(w,q))
 def r(q):
  g=d(cu,q); return (g@lin(g,cv)-cv).ravel()
 lo=[.00001,-4]; hi=[10,8]; seeds=[[.25,.15],[.85,1.2],[1.6,2.0],[2.5,-.3]]
 if k.endswith('per'):lo=[.00001]*4;hi=[10]*4;seeds=[[s[0]]*4 for s in seeds]
 if k in ['softfree','tanhabs','atan']:lo=lo[:1];hi=hi[:1];seeds=[s[:1] for s in seeds]
 if k in ['softpower','softsat','oddabsrat','softtanh']:lo[1]=.00001;seeds=seeds[:-1]
 opt=min((least_squares(r,ss,bounds=(lo,hi),gtol=1e-12,max_nfev=400) for ss in seeds),key=lambda q:q.cost); q=opt.x; c=lin(d(cu,q),cv); C[k]=(fun,q,c); print('C',k,q,np.std(opt.fun))
B=lin(one(p['panel_b']['u']),p['panel_b']['v']); Bi=lambda w:(w-B[0])@np.linalg.inv(B[1:]); Bd=lambda w:one(w)@B
h=__import__('hegel_probe');h.vi=np.arange(52,72);s=h.study(np.arange(52),h.vi);x=np.r_[s['train']['x'],s['test']['x']]; scores=[]
for mode in ['cyc','poly','polyabs']:
 c=lin(AF(u,mode),v); aa=lin(one(u),v)
 def Ai(X):
  return np.array([root(lambda z:(AF(z[None,:],mode)@c)[0]-xx,(xx-aa[0])@np.linalg.inv(aa[1:])).x for xx in X])
 for label,z in [('Ai',Ai(x)),('AiBi',Bi(Ai(x))),('AiBd',Bd(Ai(x))),('BiAi',Ai(Bi(x))),('BdAi',Ai(Bd(x)))]:
  for name,(fun,q,c1) in C.items():
   c=one(fun(z,q))@c1; d=a._panel_maps(p['panel_c'],z)[:,4:];
   for wname,w in [('cd',np.c_[c,d]),('cdz',np.c_[c,d,z]),('cz',np.c_[c,z]),('z',z),('zd',np.c_[z,d])]:
    for scale in [0,1]:
     W=w if not scale else (w-w[:52].mean(0))/np.maximum(w[:52].std(0),.1); e=h.score(W,s);scores.append((max(e),e,mode,label,name,wname,scale))
print('TOP',*sorted(scores)[:30],sep='\n')

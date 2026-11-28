exec(open('/tmp/probe.py').read().split("print('A',")[0]); from scipy.optimize import least_squares
c=s['panel_c']; u=c['u']; D=af(u); Q=poly(u,2); R=np.eye(len(u))-D@np.linalg.pinv(D); v1,v2=c['v1'],c['v2']; print('C quadratic coefficients v1/v2',reg(Q,v1).T,reg(Q,v2).T,sep='\n'); print('C quadratic RMS',np.sqrt(np.mean((Q@reg(Q,v1)-v1)**2,0)),np.sqrt(np.mean((Q@reg(Q,v2)-v2)**2,0))); print('C linear residual singular values',*[np.linalg.svd(R@v,compute_uv=False) for v in (v1,v2)])
def rms(e):return np.sqrt(np.mean(e**2))
for k,v in [('v1',v1),('v2',v2)]:
 print(k,'coordinate nonlinearities after affine removal')
 for name,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('sat',lambda z:z/(1+z*z)),('softsign',lambda z:z/np.sqrt(1+z*z))]:
  scores=[]
  for f in (.6,.88,1,1.2,1.7,2,2.4,3):
   F=np.c_[D,fun(f*u)]; H=F@np.linalg.pinv(F); e=(v-H@v)/(1-np.diag(H))[:,None];scores.append((rms(e),f))
  print(name, sorted(scores)[:2])
 for j in range(4):
  yy=v[:,j]; sols=[]
  for name,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('sat',lambda z:z/(1+z*z)),('softsign',lambda z:z/np.sqrt(1+z*z)),('sq',np.square),('cube',lambda z:z**3)]:
   def funfit(p):return R@(p[-1]*fun(D@p[:5])-yy)
   gen=np.random.default_rng(16); starts=[np.r_[0,np.eye(4)[i]*1.7,.3] for i in range(4)]+[np.r_[0,reg(D,v)[:,j][1:]*1.7,.3]]+[np.r_[gen.normal(0,.65,5),.3] for _ in range(5)]
   opt=min((least_squares(funfit,p,max_nfev=150,bounds=([-5]*5+[-4],[5]*5+[4])) for p in starts),key=lambda o:rms(o.fun));sols.append((rms(opt.fun),name,opt.x))
  print(j,sorted(sols,key=lambda a:a[0])[:3])
print('A diagonal',np.diag(w[1:]),'B',reg(af(s['panel_b']['u']),s['panel_b']['v']))

import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/probe.py').read())
from scipy.optimize import least_squares, minimize_scalar
p=s['panel_a']; u=p['u']; v=p['v']; slope=np.array([np.median((v[:,i,None]-v[:,i])[(abs(u[:,i,None]-u[:,i])>.2)]/(u[:,i,None]-u[:,i])[(abs(u[:,i,None]-u[:,i])>.2)]) for i in range(4)]); b0=np.median(v-u*slope,0)
mask=np.max(abs(v-u*slope-b0),1)<.01
coef=np.array([fit(aff(u[mask,i,None]),v[mask,i]) for i in range(4)]); z=(x-coef[:,0])/coef[:,1]
q=fit(aff(s['panel_b']['u']),s['panel_b']['v']); B=q[1:]; b=q[0]; print('B singular values',svdvals(B),'A clean',mask.sum())
D=-np.expm1(-ts[:,None]*np.array([.405,1.831])); C=fit(D,T.T).T; Y=np.c_[C,y]; print('latent corr',np.corrcoef(Y.T))
P=s['panel_c']; U=P['u']; R=P['v1']; S=P['v2']; src={'u':U,'v1':R,'Bu':aff(U)@q}; funcs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'exp':np.exp,'square':np.square,'softplus':lambda a:np.logaddexp(0,a),'gauss':lambda a:np.exp(-a*a)}
for label,V in [('v1',R),('v2',S)]:
 results=[]
 for name,W in src.items():
  if name==label:continue
  for fname,f in funcs.items():
   def design(k):return aff(np.c_[W,f(W*k)])
   for k in (.5,1,1.5,2,3):
    E=design(k); results.append((np.mean(abs(E[18:]@fit(E[:18],V[:18])-V[18:])),name,fname,k))
 print(label,sorted(results)[:12]);
 for name,W in src.items():
  if name!=label:print('rank nonlinear residual',label,name,svdvals(V-aff(W)@fit(aff(W),V)))
spaces={'z':z,'Bz':z@B+b,'invBz':(z-b)@np.linalg.inv(B),'Bx':x@B+b,'x':x}
for label,W in spaces.items():
 results=[]
 for nm,f in {**funcs,'id':lambda w:w}.items():
  for k in [.5,1,1.5,2]:
   for mode in [0,1,2]:
    E=[aff(f(W*k)),aff(np.c_[W,f(W*k)]),poly(f(W*k),2)][mode]; P=E[56:]@fit(E[:56],Y[:56]); mae=np.mean(abs(P-Y[56:]),0); err=np.mean(abs((P[:,:2]-C[56:])@D.T));results.append((err,mae[2],nm,k,mode))
 print(label,'traj',sorted(results)[:5],'y',sorted(results,key=lambda a:a[1])[:5])
 for j in range(3):
  E=aff(W); ww=fit(E,Y[:,j]); print(label,j,'linear',ww,'corr',np.corrcoef(np.c_[W,Y[:,j]].T)[-1,:4])

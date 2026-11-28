from operators import *
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; inv,B,C,D=operators(s)
z=inv(s['train']['x']);x=s['train']['x'];T=s['train']['traj'];y=s['train']['y'];t=s['eval_times'];q=lambda p:-np.expm1(-t[:,None]*p)
p=least_squares(lambda p:(q(p)@fit(q(p),T[:52].T)-T[:52].T).ravel(),[.4,1.8]).x
Y=np.c_[y,fit(q(p),T.T).T]; n=52
F={}
for k,u in {'z':z,'Bz':B(z),'Cz':C(z),'Dz':D(z),'CBz':C(B(z)),'DBz':D(B(z)),'BCz':B(C(z)),'BDz':B(D(z)),'raw':x,'edge':z*np.roll(z,-1,axis=1),'cos':np.cos(2.4*z)}.items():
 F[k]=u
 print(k)
 for name,h in [('lin',u),('poly2',np.c_[u,*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]]),('tanh',np.tanh(u)),('sin',np.sin(u)),('tanh+lin',np.c_[u,np.tanh(u)]),('lin+squares',np.c_[u,u*u])]:
  H=one(h); r=H@fit(H[:n],Y[:n])-Y; print(name,np.mean(abs(r[:n]),0).round(3),np.mean(abs(r[n:]),0).round(3))
 for j in range(3):
  h=one(u); yj=Y[:,j]; out=[]
  for name,f in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('signedrat',lambda x:x/(1+abs(x))),('rat',lambda x:x/(1+x*x)),('quad',lambda x:x+x*x)]:
   def pred(b):return b[-2]+b[-1]*f(h@b[:5])
   for sc in [.4,1.2]:
    a=fit(h[:n],yj[:n]); a[0]-=yj[:n].mean(); b0=np.r_[a/(sc+1e-8),yj[:n].mean(),sc]; opt=least_squares(lambda b:(pred(b)-yj)[:n],b0,max_nfev=120);r=pred(opt.x)-yj;out.append((np.mean(abs(r[:n])),np.mean(abs(r[n:])),name,opt.x.round(2)))
  print('idx',j,[(round(a,3),round(b,3),c,d.tolist()) for a,b,c,d in sorted(out,key=lambda x:x[0])[:2]])
# screen sparse transformed features
ff=[]; names=[]
for k,u in F.items():
 for nm,h in [('id',u),('sq',u*u),('cube',u**3),('tanh',np.tanh(u)),('sin',np.sin(u)),('cos',np.cos(u)),('rat',u/(1+u*u)),('sgnrat',u/(1+abs(u))),('exp',np.exp(-.5*u*u)),('tanh2',np.tanh(2*u))]:
  ff.extend(h.T); names.extend([k+nm+str(i) for i in range(4)])
 for i,j in combinations(range(4),2):
  a,b=u[:,i],u[:,j];
  for nm,h in [('mul',a*b),('sumtan',np.tanh(a+b)),('diftan',np.tanh(a-b)),('sumsin',np.sin(a+b)),('difsin',np.sin(a-b)),('sumcos',np.cos(a+b)),('difcos',np.cos(a-b)),('mulrat',a*b/(1+a*a+b*b)),('amul',a*a*b),('bmul',a*b*b)]: ff.append(h);names.append(k+nm+str(i)+str(j))
H=np.array(ff).T;H-=H[:n].mean(0);H/=np.maximum(H[:n].std(0),1.e-8)
for j in range(3):
 yj=Y[:,j];r=yj-yj[:n].mean();ids=[]
 for i in range(12):
  a=np.argmax(abs(H[:n].T@r[:n]));ids.append(a);h=one(H[:,ids]);r=yj-h@fit(h[:n],yj[:n]);print('omp',j,i,names[a],round(np.mean(abs(r[:n])),4),round(np.mean(abs(r[n:])),4))
print('Y prediction from true amplitudes')
for h in [Y[:,1:],np.c_[Y[:,1:],Y[:,1:]**2,Y[:,1]*Y[:,2]],np.c_[F['Cz'],F['Dz'],Y[:,1:]]]:
 h=one(h);r=h@fit(h[:n],y[:n])-y; print(np.mean(abs(r[:n])),np.mean(abs(r[n:])))

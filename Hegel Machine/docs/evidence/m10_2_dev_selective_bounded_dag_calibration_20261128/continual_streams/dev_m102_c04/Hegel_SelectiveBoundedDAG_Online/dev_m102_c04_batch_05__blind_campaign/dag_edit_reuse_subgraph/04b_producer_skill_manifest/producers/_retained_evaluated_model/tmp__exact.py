exec(open('/tmp/probe.py').read().split("print('telemetry'")[0])
from itertools import combinations_with_replacement as cwr
u=s['panel_a']['u'];v=s['panel_a']['v']; _,_,m=clip(af(u),v);u,v=u[m],v[m]
a,b=u.ravel(),np.roll(u,-1,axis=1).ravel();y=v.ravel();d=np.tile(np.eye(4),(len(u),1));p=list(cwr(range(4),2));t=list(cwr(range(4),3));q=lambda u,p:np.column_stack([np.prod(u[:,j],axis=1) for j in p]);
for j in range(4):
 x=np.roll(u,-j,axis=1); f=np.c_[af(x),q(x,p),q(x,t)];w=ls(f,v[:,j]);print('a cubic',j,np.linalg.norm(f@w-v[:,j]),[(name,round(i,5)) for name,i in zip(['1',0,1,2,3]+p+t,w) if abs(i)>.009]);
for k in [2,3,4,5,6]:
 x=np.c_[a,b]; terms=[t for l in range(1,k+1) for t in cwr(range(2),l)];f=np.c_[d,q(x,terms)];w=ls(f,y);print('pooled',k,np.linalg.norm(f@w-y),[(name,round(i,5)) for name,i in zip(list(range(4))+terms,w) if abs(i)>.003]);
for k in [2,3,4]:
 f=np.c_[d,d*a[:,None],d*b[:,None],d*(a*b)[:,None],d*(a*a)[:,None],d*(b*b)[:,None],*[d*(a**i*b**(l-i))[:,None] for l in range(3,k+1) for i in range(l+1)]];w=ls(f,y);print('a per local',k,np.linalg.norm(f@w-y),w.reshape(-1,4))
u=s['panel_c']['u'];v=s['panel_c']['v2']; z=u*u
for name,fun,init,lim in [('rational4',lambda k:z/(1+k*z),np.ones(4),(0.,10.)),('rationalpower',lambda k:z/(1+k[:4]*z**k[4]),[1]*5,(0.,10.)),('exp4',lambda k:-np.expm1(-k*z),np.ones(4),(0.0001,10.)),('tanh4',lambda k:np.tanh(k*z),np.ones(4),(0.0001,10.)),('sqrt4',lambda k:1/np.sqrt(1+k*z),np.ones(4),(0.0001,10.)),('cos4',lambda k:np.cos(k*u),np.ones(4),(0.0001,10.)),('gaussian4',lambda k:np.exp(-k*z),np.ones(4),(0.0001,10.)),('invqu4',lambda k:z/(1+k*z*z),np.ones(4),(0.,10.))]:
 def f(k):return af(fun(np.asarray(k)))
 opt=least_squares(lambda k:(f(k)@ls(f(k),v)-v).ravel(),init,bounds=lim);print(name,opt.x,np.linalg.norm(opt.fun))

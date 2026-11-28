import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=s['panel_b']['u'],s['panel_b']['v'];a=s['panel_a'];p,q=a['u'],a['v'];ok=np.max(abs(q-p-np.median(q-p,0)),1)<1;b,g=np.array([np.linalg.lstsq(np.c_[np.ones(sum(ok)),p[ok,i]],q[ok,i],rcond=None)[0] for i in range(4)]).T
z=(s['train']['x']-b)/g;Y=np.c_[s['train']['y'],s['train']['traj']];aff=lambda x:np.c_[np.ones(len(x)),x]; n=len(u)
fs={}
for k in (1,2,3,4,5,6,8):
 for typ in ('rat','exp','root','ratlin','explin','rootlin'):
  fs[typ+str(k)]=(lambda x,p,k=k,typ=typ: ((x**3+(p[1]*x if typ.endswith('lin') else 0))*({'rat':lambda a:1/(1+a),'exp':lambda a:np.exp(-a),'root':lambda a:1/np.sqrt(1+a)}[typ.removesuffix('lin')])(p[0]*abs(x)**k)), [.22,0] if typ.endswith('lin') else [.22], [0,-1] if typ.endswith('lin') else [0], [5,1] if typ.endswith('lin') else [5])
fs.update({'tanhcub':(lambda x,p:np.tanh(p[0]*x**3),[.4],[.001],[5]), 'sincub':(lambda x,p:np.sin(p[0]*x**3),[.4],[.001],[2]),'atancub':(lambda x,p:np.arctan(p[0]*x**3),[.4],[.001],[5]),'cubicshift':(lambda x,p:(x+p[0])**3,[0],[-.5],[.5]),'cubictau':(lambda x,p:x**3/(1+p[0]*abs(x)**p[1]),[.22,4],[0,.1],[5,16]),'expfree':(lambda x,p:x**3*np.exp(-p[0]*abs(x)**p[1]),[.22,4],[0,.1],[5,16]),'tanhprod':(lambda x,p:x*x*np.tanh(p[0]*x),[1.25],[.001],[10]),'shiftpoly5':(lambda x,p:(x+p[2])**3+p[0]*(x+p[2])**5+p[1]*(x+p[2])**7,[-.22,.02,0],[-3,-3,-.5],[3,3,.5]),'powered':(lambda x,p:np.sign(x)*abs(x)**p[1]/(1+p[0]*abs(x)**p[2]),[.22,3,4],[0,.5,.1],[5,7,16]),'sinwarp':(lambda x,p:np.sin(p[0]*x)**3,[.6],[.001],[2.5])})
res=[]
for name,(f,p,lo,hi) in fs.items():
 def fun(p):
  A=aff(f(u,p));return (v-A@np.linalg.lstsq(A,v,rcond=None)[0]).ravel()
 r=least_squares(fun,p,bounds=(lo,hi),max_nfev=120,gtol=1e-12);A=aff(f(u,r.x));P=np.linalg.pinv(A); e=v-A@(P@v);h=(A*P.T).sum(1);w=aff(f(z,r.x))@(P@v);i,j=np.triu_indices(4); W=aff(np.c_[w,w[:,i]*w[:,j]]);pred=W[52:]@np.linalg.lstsq(W[:52],Y[:52],rcond=None)[0]; err=abs(pred-Y[52:]);res.append((np.sqrt(np.mean((e/(1-h[:,None]))**2)),name,r.x,np.sqrt(np.mean(e*e)),err[:,0].mean(),err[:,1:].mean(),err[:,1:].mean(0).max(),f,P@v))
for q in sorted(res,key=lambda q:q[0])[:30]:print(q[:-2]);
for q in sorted(res,key=lambda q:q[0])[:4]:
 f,B=q[-2:];w=aff(f(z,q[2]))@B; i,j=np.triu_indices(4); W=aff(np.c_[w,w[:,i]*w[:,j]]);D=np.linalg.lstsq(W[:52],Y[:52],rcond=None)[0]; print(q[1],'coeff',np.round(D[:,[0,1,-1]],4),'train RMSE',np.sqrt(np.mean((W[:52]@D-Y[:52])**2,0)))

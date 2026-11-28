import pickle,numpy as np
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
poly=lambda x,d:PolynomialFeatures(d).fit_transform(x)
p=s['panel_a']; u,v=p['u'],p['v']; m=np.linalg.norm(u-v,axis=1)<1
A=np.array([ls(poly(u[m,i:i+1],1),v[m,i]) for i in range(4)]); af=lambda x:x*A[:,1]+A[:,0]; ai=lambda x:(x-A[:,0])/A[:,1]
p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); bf=lambda x:poly(x,1)@B
from scipy.optimize import least_squares
from itertools import combinations
from scipy.linalg import qr
p=s['panel_c']; u=p['u']; one=lambda q:np.ones((len(q),1)); hh=lambda q,par:q*np.exp(par[1]*np.log1p(par[0]*abs(q)**par[2])); funs={'soft':(lambda q,k:q/np.sqrt(1+k*q*q),[3.5],([.02],[50])),'tanh':(lambda q,k:np.tanh(k*q),[2],([.02],[20])),'atan':(lambda q,k:np.arctan(k*q),[2],([.02],[20])),'cos':(lambda q,k:np.cos(k*q),[1.25],([.02],[8])),'exp':(lambda q,k:np.exp(-k*q*q),[.75],([.02],[8])),'general':(lambda q,*k:hh(q,k),[3.5,-.5,2],([.02,-5,.25],[50,3,7])),'logcosh':(lambda q,k:np.logaddexp(k*q,-k*q),[1],([.02],[8])),'sin':(lambda q,k:np.sin(k*q),[1.25],([.02],[8]))}; models=[]
for key in ('v1','v2'):
 v=p[key]; fit=[]
 for n,(f,init,bounds) in funs.items():
  for extra in (0,1,2):
   def D(q,k): return np.c_[one(q),*[q] * (extra>=1),*[q*q] * (extra>=2), f(q,*k)]
   opt=least_squares(lambda k:(D(u,k)@ls(D(u,k),v)-v).ravel(),init,bounds=bounds,ftol=1e-12,gtol=1e-12); k=opt.x; H=D(u,k); coef=ls(H,v); err=np.mean(abs(H@coef-v)); fit.append((err,n,extra,k,coef))
 fit.sort(key=lambda z:z[0]); print(key,[(round(e,7),n,d,k.round(5)) for e,n,d,k,c in fit[:10]]); print('top coefficients',fit[0][-1].round(5)); print('simple coefficients',min([z for z in fit if z[0]<max(0.00014,fit[0][0]*1.15)],key=lambda z:z[2]+len(z[3]))[-1].round(4)); best=min([z for z in fit if z[0]<max(0.00014,fit[0][0]*1.15)],key=lambda z:z[2]+len(z[3])); e,n,d,k,c=best; models.append(lambda q,n=n,d=d,k=k,c=c:np.c_[one(q),*[q]*(d>=1),*[q*q]*(d>=2),funs[n][0](q,*k)]@c)
splits=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=12).split(x)]; L=lambda x:poly(x,1); Q=lambda x:poly(x,2)
def score(F):
 e=[np.mean(abs(F[b]@ls(F[a],y[a])-y[b])) for a,b in splits]; return tuple(np.round([e[0],np.mean(e[1:]),max(e[1:]),np.std(y-F@ls(F,y))],5))
coords={'raw':x,'invA':ai(x),'A':af(x),'BA':bf(af(x)),'BinvA':bf(ai(x)),'Braw':bf(x),'invBA':(af(x)-B[0])@np.linalg.inv(B[1:]),'invBinvA':(ai(x)-B[0])@np.linalg.inv(B[1:])}; results=[]
for nm,q in coords.items():
 v,w=[f(q) for f in models]; branch={'v':v,'w':w,'vw':v*w,'vpw':v+w,'vmw':v-w,'v2':v*v,'w2':w*w,'all':np.c_[v,w],'v/w':v/(1+.22*w*w),'w/v':w/(1+.22*v*v),'vw2':v*w*w,'wv2':w*v*v,'tanhv':np.tanh(1.25*v),'tanhw':np.tanh(1.25*w),'w-v2':w-v*v}
 for name,z in branch.items():
  for dn,F in [('lin',L(z)),('quad',Q(z)),('xlin',np.c_[L(q),z]),('xquad',np.c_[Q(q),z]),('xquadF',np.c_[Q(q),Q(z)[:,1:]])]:
   if F.shape[1]<49: results.append((score(F),nm,name,dn))
 for i in range(4):
  for g in [.22,.8,1.25,2,4]:
   for name,z in [('soft',q[:,i]/np.sqrt(1+g*q[:,i]**2)),('cos',np.cos(g*q[:,i])),('exp',np.exp(-g*q[:,i]**2)),('rat',1/(1+g*q[:,i]**2))]:
    for typ,G in [('L',L(q)),('Q',Q(q)),('C',np.c_[L(q),v,w])]: results.append((score(np.c_[Q(q),z[:,None]*G]),nm,i,g,name,typ))
print('best',*sorted(results,key=lambda r:max(r[0][0],r[0][2]))[:25],sep='\n'); print('split',*sorted(results,key=lambda r:r[0][0])[:10],sep='\n')

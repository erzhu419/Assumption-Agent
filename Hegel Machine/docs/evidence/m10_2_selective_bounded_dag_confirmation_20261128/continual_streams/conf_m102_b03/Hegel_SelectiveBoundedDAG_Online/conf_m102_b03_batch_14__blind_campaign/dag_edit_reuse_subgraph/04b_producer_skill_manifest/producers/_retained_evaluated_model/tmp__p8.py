import numpy as np,pickle,itertools,runpy
m=runpy.run_path('/holdout_task/node_00.py');s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=m['run_node']({'__input__':s})['z'];print('z range',z.min(0),z.max(0));p=s['panel_a']; zz=m['_calibrate'](p,p['v']);print('alignment inversion median/max',np.median(abs(zz-p['u']),axis=0),np.max(abs(zz-p['u']),axis=0));u=s['panel_c']['u']; o=np.ones(len(u)); i,j=np.array([(0,1),(0,3),(1,2),(2,3)]).T; p=u[:,i]*u[:,j];
for k in (1,2):
 v=s['panel_c']['v'+str(k)]; forms={'raw':u,'pair':p,'square':u*u,'tanh':np.tanh(1.5*u),'rat':u/(1+.86*u*u),'cos':np.cos(2.4*u),'soft':u/(1+.85*abs(u)),'sin':np.sin(2.94*u),'rat2':u*u/(1+.165*u*u),'pairrat':p/(1+.85*p*p),'pairsq':p*p,'paircos':np.cos(2.4*p),'triples':np.column_stack([np.prod(u[:,t],axis=1) for t in itertools.combinations(range(4),3)])};out=[]
 for n in (1,2,3):
  for tags in itertools.combinations(forms,n):
   A=np.column_stack((o,*(forms[t] for t in tags))); P=np.linalg.pinv(A); E=A@(P@v)-v; H=(A*P.T).sum(1);out.append((np.mean((E/(1-H[:,None]))**2)**.5,np.mean(E**2)**.5,tags))
 print('branch',k,*sorted(out)[:14],sep='\n');best=sorted(out)[0]; A=np.column_stack((o,*(forms[t] for t in best[2])));print('coeff',np.round(np.linalg.lstsq(A,v,rcond=None)[0],3));print('row resid',np.round(np.linalg.norm(A@np.linalg.lstsq(A,v,rcond=None)[0]-v,axis=1),4))

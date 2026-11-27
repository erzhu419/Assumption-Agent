import itertools,pickle,numpy as np
from scipy.sparse import diags,kronsum,eye
from scipy.sparse.linalg import splu
c=pickle.load(open('/holdout_task/public_beta_m96_w02_s00.pkl','rb'))[0]; s,g=c['study'],c['gt']; V=s['V_reaction_2d']; T=g['S_tilde']; nx,ny=V.shape; i,j=s['sensor_ij'].T; Z=s['Z_nuisance']; P,d,_=np.linalg.svd(Z,full_matrices=False); r=s['nuisance_ridge']; y=s['y_obs_raw']; src=s['source_modes'].reshape(len(g['w_opt']),-1).T
print('target/profile errors',[(e,np.max(abs(y-P@((1-(r/(d*d+r))**e)*(P.T@y))-g['r_y']))) for e in (0,.5,1)]); print('target baseline norms',np.linalg.norm(T),np.linalg.norm(g['r_y']))
def axis(a,style,boundary,side):
 k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/len(k); b=s['robin_'+a]; D=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2; A=np.diag(np.r_[0,D]+np.r_[D,0])+np.diag(-D,1)+np.diag(-D,-1); A[0,0]+=b*k[0]/(h*(b*.5*h+k[0])); A[-1,-1]+=b*k[-1]/(h*(b*.5*h+k[-1])); v=v*side/h
 if style=='cell':
  A+=np.diag(abs(v)); A+=np.diag(-np.maximum(v[1:],0),-1)+np.diag(np.minimum(v[:-1],0),1)
  if not boundary: A[0,0]-=max(v[0],0); A[-1,-1]+=min(v[-1],0)
 else:
  f=(v[:-1]+v[1:])/2 if style=='mean' else (v[:-1] if style=='left' else v[1:]); p=np.maximum(f,0); m=np.minimum(f,0); A+=np.diag(np.r_[p,0]-np.r_[0,m])+np.diag(m,1)+np.diag(-p,-1)
  if boundary: A[0,0]-=min(v[0],0); A[-1,-1]+=max(v[-1],0)
 return diags([A.diagonal(-1),A.diagonal(),A.diagonal(1)],[-1,0,1],format='csc')
results=[]
for style,boundary,side,pot in itertools.product(('cell','mean','left','right'),(0,1),(1,-1),('V','R','VR')):
 R=s['reaction_x'][:,None]+s['reaction_y'][None,:]; D={'V':V,'R':R,'VR':V+R}[pot]; A=kronsum(axis('y',style,boundary,side),axis('x',style,boundary,side),format='csc')+diags(D.ravel()); vals=[]
 for mu in s['mu_shifts']:
  U=splu(A+mu*eye(nx*ny,format='csc')).solve(src).reshape(nx,ny,-1); vals.append(U[i,j]);
 err_field=np.linalg.norm(U@g['w_opt']-g['terminal_field_A'])/np.linalg.norm(g['terminal_field_A']); vals=np.array(vals)
 for order in (0,1):
  F=(vals if order==0 else vals.transpose(1,0,2)).reshape(T.shape)
  for e in (.5,1,0):
   FT=F-P@((1-(r/(d*d+r))**e)[:,None]*(P.T@F)); err=np.linalg.norm(FT-T)/np.linalg.norm(T); results.append((err,err_field,style,boundary,side,pot,order,e,np.max(abs(FT-T))))
print('best design:'); print(*sorted(results)[:10],sep='\n'); print('best field:'); print(*sorted(results,key=lambda x:x[1])[:6],sep='\n')

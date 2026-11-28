exec(open('/tmp/discover.py').read().split('A=old._calibration')[0]);from scipy.optimize import least_squares
A=old._calibration(s['panel_a']);z=old._decode(x,A);pb=s['panel_b'];B=fit(poly(pb['u'],1),pb['v']);w=poly(z,1)@B;pc=s['panel_c'];U=pc['u'];V1=pc['v1'];V2=pc['v2'];print('telemetry',s['telemetry_vec'],'summary',s['train']['summary']);D=lambda z:poly(z,1)
for j,V in enumerate((V1,V2)):
 for nm,fn,p0 in [('affcos',lambda x,p:np.c_[x,np.cos(p[0]*x)],[1.3]),('atan+sin',lambda x,p:np.c_[np.arctan(p[0]*x),np.sin(p[1]*x)],[4.4,2.58]),('tanh+sin',lambda x,p:np.c_[np.tanh(p[0]*x),np.sin(p[1]*x)],[4.4,2.6]),('tanh+sin+aff',lambda x,p:np.c_[x,np.tanh(p[0]*x),np.sin(p[1]*x)],[4.4,2.6]),('atan+sin+aff',lambda x,p:np.c_[x,np.arctan(p[0]*x),np.sin(p[1]*x)],[4.4,2.58]),('atan+sin same',lambda x,p:np.arctan(p[0]*x)+p[2]*np.sin(p[1]*x),[4.4,2.58,0.5]),('tanh+sin same',lambda x,p:np.tanh(p[0]*x)+p[2]*np.sin(p[1]*x),[4.4,2.6,0.5]),('tanh of sin',lambda x,p:np.tanh(p[0]*x+p[2]*np.sin(p[1]*x)),[1.0,2.6,0.5]),('atan of sin',lambda x,p:np.arctan(p[0]*x+p[2]*np.sin(p[1]*x)),[1.0,2.6,0.5]),('tanh+sin+aff same',lambda x,p:np.tanh(p[0]*x)+p[2]*np.sin(p[1]*x)+p[3]*x,[4.4,2.6,0.5,0.1]),('cos+aff same',lambda x,p:x+p[1]*np.cos(p[0]*x),[1.3,-.65])]:
  if (j==1) != ('cos' in nm): continue
  def P(x,p):return np.c_[np.ones(len(x)),fn(x,p)]
  def res(p):F=P(U,p);return (F@fit(F,V)-V).ravel()
  r=least_squares(res,p0,max_nfev=1500); C=fit(P(U,r.x),V);print('panel',j,nm,r.x,'rms',np.std(r.fun));print(C.round(4));
  if nm=='atan+sin+aff':a=P(z,r.x)@C
  if nm=='cos+aff same':b=P(z,r.x)@C
H=np.c_[D(z),a,b,a*b,a*a,b*b]; names=['const']+[f'{s}{i}' for s in ['z','a','b','ab','aa','bb'] for i in range(4)];
cut=np.arange(52); print('full coeffs',list(zip(names,fit(H[cut],y[cut]).round(5))));print('condition',np.linalg.cond(H[cut]));
print('traj summary', t.mean(0),t.std(0));print('traj linear normalized coefficients',fit(D(z)[:52],t[:52]));
for design,lab in [(H,'25 pairwise'),(np.c_[D(z),a,b,poly(np.c_[a,b],2)[:,9:]],'all pairwise')]:
 n=design.shape[1]; perm=np.random.default_rng(29).permutation(72); inds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),perm[i:i+18]),perm[i:i+18]) for i in range(0,72,18)];
 for lam in [0,1e-5,1e-4,1e-3,.01,.1]:
  mae=[]
  for i,j in inds:
   X=design[i]; scale=np.r_[1.,X[:,1:].std(0).clip(1e-8)];Z=X/scale;rr=np.eye(n)*lam;rr[0,0]=0; c=np.linalg.solve(Z.T@Z+rr,Z.T@y[i]) if lam else fit(Z,y[i]);mae.append(np.mean(abs(design[j]/scale@c-y[j])));
  print(lab,lam,np.round(mae,5))

exec(open('/tmp/probe.py').read().split("print('traj sv'")[0])
P=S['panel_a']; u,v=P['u'],P['v']; print('A norms',np.sort(np.linalg.norm(v,axis=1)).round(2))
def des(u,j): return np.c_[af(u),u[:,j]*u[:,(j+1)%4],u[:,j]**3]
def calibrate(x):
 A=[des(u,j) for j in range(4)]; good=np.ones(len(u),bool)
 for _ in range(8):
  co=np.column_stack([fit(a[good],v[good,j]) for j,a in enumerate(A)])
  r=np.linalg.norm(np.column_stack([a@co[:,j] for j,a in enumerate(A)])-v,axis=1); good=r<max(.004,2.8*np.median(r))
 print('A residual',np.sort(r).round(4));print('A co',co.round(5))
 def forward(w):return np.array([(des(w[None],j)@co[:,j])[0] for j in range(4)])
 return np.array([least_squares(lambda w:forward(w)-a,b,gtol=1e-12).x for a,b in zip(x,np.linalg.solve(co[1:5].T,(x-co[0]).T).T)])
z=calibrate(x); B=S['panel_b']; b=poly(z,3)@fit(poly(B['u'],3),B['v'])
cb=fit(af(B['u']),B['v']); b=af(z)@cb
C=S['panel_c']; u=C['u']; fn={k:f for k,f in [('lin',lambda w:w),('atan',lambda w:np.arctan(2.15*w)),('atan125',lambda w:np.arctan(1.25*w)),('rat',lambda w:w/(1+.22*w*w)),('tanh',lambda w:np.tanh(1.25*w)),('sin',lambda w:np.sin(2.15*w)),('sq',lambda w:w*w)]}
def v1(w,h):return np.arctan(h[0]*w)+h[1]*w/(1+h[2]*w*w)
for F in (lambda w,h:np.arctan(h[0]*w),v1):
 def res(h):
  A=af(F(u,h));return (A@fit(A,C['v1'])-C['v1']).ravel()
 h=least_squares(res,[2.15] if F!=v1 else [2.15,.1,.22],max_nfev=500).x;print('C fit',h,np.std(res(h)))
 fn['v1']=lambda w:af(F(w,h))@fit(af(F(u,h)),C['v1'])
fn['v2']=lambda w:np.c_[af(w),w*w,w**3]@fit(np.c_[af(u),u*u,u**3],C['v2'])
for label,w in [('z',z),('b',b)]:
 for k,F in fn.items():
  a=F(w);score(label+k,af(a));score(label+k+'2',poly(a,2));score(label+k+'+z',af(np.c_[z,a]));
  if k in ['atan','v1','v2']:
   score(label+k+'2+z',poly(np.c_[z,a],2))

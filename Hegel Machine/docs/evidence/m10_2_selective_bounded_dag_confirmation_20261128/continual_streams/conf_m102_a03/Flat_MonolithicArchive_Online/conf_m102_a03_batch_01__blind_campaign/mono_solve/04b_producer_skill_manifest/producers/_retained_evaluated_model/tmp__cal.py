import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return PolynomialFeatures(d).fit_transform(x)
u=s['panel_a']['u']; v=s['panel_a']['v']; good=np.linalg.norm(v-u,axis=1)<3; print('excluded',np.flatnonzero(~good))
u=u[good];v=v[good]; a=u.ravel();b=np.roll(u,-1,axis=1).ravel();ch=np.tile(np.eye(4),(len(u),1));
S=np.c_[ch,a*1.,b,a*b,a**3,b**3,a*a*b,a*b*b,a*a,b*b,a**5,b**5]
Y=(v-u).ravel();
for stop in (7,8,11,13,15):
 C=fit(S[:,:stop],Y); print('shared',stop,C,'rmse',np.sqrt(np.mean((S[:,:stop]@C-Y)**2)))
for j in range(4):
 a,b=u[:,j],u[:,(j+1)%4]; w=v[:,j]; D=poly(np.c_[a,b],4)
 for n in (10,15):
  C=fit(D[:,:n],w); print('per channel',j,n,C,'rmse',np.std(D[:,:n]@C-w))
for d in (2,3,4):
 for j in (0,):
  P=PolynomialFeatures(d);D=P.fit_transform(u);C=fit(D,v[:,j]); print('full',d,'rmse',np.std(D@C-v[:,j]),list(zip(P.get_feature_names_out()[np.argsort(abs(C))[-13:][::-1]],np.sort(abs(C))[-13:][::-1])))
# direct separable calibration; common candidate polynomials and elementary nonlinearities
forms={'cubic':lambda a,b:np.c_[np.ones(a.size),a.ravel(),b.ravel(),(a*b).ravel(),a.ravel()**3], 'septic':lambda a,b:np.c_[np.ones(a.size),a.ravel(),b.ravel(),(a*b).ravel(),a.ravel()**3,a.ravel()**5,a.ravel()**7], 'tanh':lambda a,b:np.c_[np.ones(a.size),a.ravel(),b.ravel(),(a*b).ravel(),np.tanh(a.ravel())]}
tr=s['train']; Y=np.c_[tr['y'],tr['traj']]
for name,F in forms.items():
 c=np.array([fit(F(u[:,j],u[:,(j+1)%4]),v[:,j]) for j in range(4)])
 def forward(z):return np.column_stack([F(z[:,j],z[:,(j+1)%4])@c[j] for j in range(4)])
 print(name,'coef',c,'cal rmse',np.std(forward(u)-v)); z=np.stack([least_squares(lambda z:forward(z[None])[0]-x,x).x for x in tr['x']]); print('z range',z.min(0),z.max(0),'inversion',np.max(abs(forward(z)-tr['x'])))
 for d in (1,2,3):
  D=poly(z,d); r=D@fit(D[:52],Y[:52])-Y; print(name,d,np.mean(abs(r[:52]),axis=0)[[0,1,-1]],np.mean(abs(r[52:]),axis=0)[[0,1,-1]],np.mean(abs(r[52:,1:])))

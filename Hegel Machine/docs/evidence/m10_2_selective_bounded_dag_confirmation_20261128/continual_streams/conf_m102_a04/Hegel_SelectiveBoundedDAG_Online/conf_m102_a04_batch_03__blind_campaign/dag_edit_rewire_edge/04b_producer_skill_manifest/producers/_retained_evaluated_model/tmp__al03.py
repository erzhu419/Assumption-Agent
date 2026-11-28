exec(open('/tmp/inspect03.py').read().split("print('times'")[0])
u,v=[s['panel_a'][k] for k in ['u','v']]; keep=np.max(abs(v-u),axis=1)<3; print('kept',sum(keep));
for name,F in [('quadratic',PolynomialFeatures(2).fit_transform(u)),('odd powers',np.c_[np.ones(len(u)),u,u**2,u**3,np.tanh(u),u/(1+u*u)]),('cubic',PolynomialFeatures(3).fit_transform(u)),('abs',np.c_[np.ones(len(u)),u,u*abs(u),abs(u)])]:
 M=np.linalg.lstsq(F[keep],v[keep],rcond=None)[0]; print(name,np.mean(abs((F@M-v)[keep]),0));
 if name=='quadratic': print(PolynomialFeatures(2).fit(u).get_feature_names_out(),M)
print('A residual sample',np.c_[u[keep][:18],(v-u)[keep][:18]]); print('telemetry',s['telemetry_vec'])

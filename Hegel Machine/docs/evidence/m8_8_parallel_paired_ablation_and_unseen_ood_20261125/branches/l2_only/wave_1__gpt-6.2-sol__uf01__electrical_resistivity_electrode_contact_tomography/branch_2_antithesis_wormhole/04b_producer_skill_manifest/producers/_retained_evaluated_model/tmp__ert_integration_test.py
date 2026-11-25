from pathlib import Path
p = Path('/tmp/ert_posterior.py').read_text()
a = p.index('        if posterior:')
b = p.index('        surface =',a)
p = p[:a]+'''        if posterior:
            # Resolve the narrow depth likelihood independently of grid spacing.
            x0, z0 = previous * [self.length, self.depth]
            k, a, b, t, r = self.kernel(x0,z0)
            pk = self.project(k)
            amp = max(0.,pk @ self.y)/max(pk @ pk,1e-30)
            J = np.column_stack((pk, self.project(amp*k*((self.length-x0)/b-x0/a)),
                                 self.project(-amp*k*(t/a+r/b))))
            sse0 = self.objective(previous)[0]*self.scale
            dof = max(1,self.y.size-self.q.shape[1]-3)
            cov = np.linalg.pinv(J.T@J,rcond=1e-13)*sse0/dof
            if not (0.05 < cov[1,1]**.5 < 0.50*self.length):
                return previous
            # Adapt the depth band to the measured joint uncertainty.
            lo, hi = max(0.,z0-9*cov[2,2]**.5), min(self.depth,z0+9*cov[2,2]**.5)
            if hi-lo < 1.e-6:
                return previous
            xx=np.linspace(0.,self.length,181)
            zz=np.linspace(lo,hi,145)
            # Expand the depth band around the linearized likelihood ridge.
            zz=np.clip(zz[None,:]+np.clip(cov[1,2]/cov[1,1]*(xx[:,None]-x0),-(hi-lo)/2,(hi-lo)/2),0.,self.depth)
            weight=np.empty_like(zz)
            for start in range(0,len(xx),12):
                block = slice(start,start+12)
                zs = zz[block]
                x = np.broadcast_to(xx[block,None],zs.shape).ravel()[:,None]
                k = self.kernel(x,zs.ravel()[:,None])[0]
                qk=np.einsum('ij,jk->ik',k,self.q)
                ky=np.einsum('ij,j->i',k,self.y)
                norm2=np.einsum('ij,ij->i',k,k)-np.einsum('ij,ij->i',qk,qk)
                sse=np.maximum(self.scale-np.maximum(ky,0.)**2/np.maximum(norm2,1e-30),1e-30)
                weight[block]=np.exp(np.clip(-.5*dof*np.log(sse/max(sse0,1e-30)),-745.,300.)).reshape(zs.shape)
            slice_w=np.trapz(weight,zz,axis=1)
            norm=np.trapz(slice_w,xx)
            if norm == 0.:
                return previous
            return np.array([np.trapz(slice_w*xx,xx)/self.length,
                             np.trapz(np.trapz(weight*zz,zz,axis=1),xx)/self.depth])/norm
''' + p[b:]
p=p.replace('profile.search(radius/2, posterior=True)','profile.search(radius/2, previous=u_ml, posterior=True)')
Path('/tmp/ert_integrated.py').write_text(p)
s=Path('/tmp/ert_uncertainty_stress.py').read_text().replace('from ert_posterior import','from ert_integrated import').replace("('posterior', recovered)","('integrated', recovered)")
s=s.replace("'max',np.round(a.max(axis=0),5)","'max',np.round(a.max(axis=0),5),'rms',np.round(np.sqrt(np.mean(a[:,:2]**2,axis=0)),5)")
Path('/tmp/ert_integrated_stress.py').write_text(s)

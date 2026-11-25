import pickle, time
import numpy as np
from ert_solver import invert_ert_anomaly

cases = pickle.load(open('/holdout_task/public_ert_surveys.pkl','rb'))
for jj, item in enumerate(cases):
    survey, gt = item['survey'], item['gt']
    worst = np.zeros(2)
    labels = ['', '']
    for typ in ('full', 'tx', 'rx'):
        worst[:] = 0
        for i in range(1 if typ == 'full' else len(survey['z_'+typ+'_m'])):
            s = survey.copy()
            truth_rx = np.asarray(gt['rx_contact_offset_v']).copy()
            if typ != 'full':
                s['z_'+typ+'_m'] = np.delete(s['z_'+typ+'_m'], i)
                s['V_obs'] = np.delete(s['V_obs'], i, axis=typ=='rx')
                if typ=='rx':
                    truth_rx = np.delete(truth_rx, i)
                    truth_rx -= truth_rx.mean()
            start = time.perf_counter()
            a = invert_ert_anomaly(s)
            elapsed = time.perf_counter()-start
            assert len(a['rx_contact_offset_v']) == len(truth_rx)
            assert abs(sum(a['rx_contact_offset_v'])) < 1e-12
            err = np.array([np.linalg.norm(np.array(a['anomaly_xz_m'])-gt['anomaly_xz_m']), np.sqrt(np.mean((a['rx_contact_offset_v']-truth_rx)**2))])
            for l in range(2):
                if err[l] > worst[l]: labels[l]=str(i)
            worst = np.maximum(worst, err)
            if typ=='full': print(jj, 'prediction', a['anomaly_xz_m'], 'seconds', elapsed)
        print(jj, typ, 'worst errors', worst, 'at electrode', labels)

"""Post-hoc reading of already paid fresh rows; never a learner resource or feedback."""
def confirmation_diagnostics(report, position, tolerance=1e-7):
    from hegel_machine.cognition.evaluation import predict
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Observation
    plans = {event['probe_id']:event for event in report['actions']
        if event['action']=='plan_condition_confirmation'}
    comparisons = {event['probe_id']:event for event in report['actions']
        if event['action']=='compare_method_followup_on_fresh_response' and event.get('probe_id')}
    batches = {batch['probe']['probe_id']:batch for batch in report['batches']}
    probes = list(dict.fromkeys([*comparisons, *(key for key,event in plans.items()
        if event.get('confirmation_pair'))]))
    result = []
    for probe in probes:
        plan, comparison = plans.get(probe,{}), comparisons.get(probe)
        pair = plan.get('confirmation_pair') or comparison
        entry = {'position':position, 'probe_id':probe,
            'pair_plan':bool(plan.get('confirmation_pair')),
            'paid_phase_recorded':comparison is not None,
            'expected_disagreement':plan.get('expected_disagreement'),
            'duration_disagreement':plan.get('duration_disagreement'),
            'boundary_disagreement':plan.get('boundary_disagreement'),
            'startup_disagreement':plan.get('startup_disagreement'),
            'sampling_basis':plan.get('sampling_basis'),
            'planner_candidate_count':plan.get('planner_candidate_count'),
            'background_disagreement':plan.get('background_disagreement'),
            'interval_exposure':plan.get('interval_exposure'),
            'forecast_length':plan.get('forecast_length'),
            'basis':'posthoc frozen-program predictions on matching already-paid fresh rows'}
        batch = batches.get(probe)
        if batch is None:
            entry.update(acquired_rows=0, diagnostic_predictions=0)
            result.append(entry)
            continue
        if not pair or 'baseline_program' not in pair or 'program' not in pair:
            entry.update(acquired_rows=len(batch['observations']), diagnostic_predictions=0,
                         prediction_evidence='program pair unavailable')
            result.append(entry)
            continue
        old, new = Program.from_dict(pair['baseline_program']), Program.from_dict(pair['program'])
        rows, history = [], []
        for value in batch['observations']:
            rows.append(Observation(value['features'],value['target'],tuple(value.get('history',history))))
            history.append({**value['features'],'__target__':value['target']})
        old_predictions = [predict(old,row,pair['baseline_fallback']) for row in rows]
        new_predictions = [predict(new,row,pair['fallback']) for row in rows]
        disagreements = [(a-b)**2 for a,b in zip(old_predictions,new_predictions)]
        old_loss = sum((a-row.target)**2 for a,row in zip(old_predictions,rows))/len(rows)
        new_loss = sum((a-row.target)**2 for a,row in zip(new_predictions,rows))/len(rows)
        entry.update(acquired_rows=len(rows), diagnostic_predictions=2*len(rows),
            programs_differ=pair['baseline_program']!=pair['program'] or pair['baseline_fallback']!=pair['fallback'],
            exactly_different_prediction_rows=sum(a!=b for a,b in zip(old_predictions,new_predictions)),
            discriminating_rows=sum(value>tolerance for value in disagreements),
            mean_prediction_disagreement=sum(disagreements)/len(rows),
            maximum_prediction_disagreement=max(disagreements),
            old_fresh_loss=old_loss, new_fresh_loss=new_loss, fresh_gain=old_loss-new_loss,
            common_mismatch=old_loss>tolerance and new_loss>tolerance,
            revised_response_refuted=new_loss>tolerance,
            gain_positive=old_loss-new_loss>tolerance,
            gain_negative=new_loss-old_loss>tolerance)
        result.append(entry)
    return result

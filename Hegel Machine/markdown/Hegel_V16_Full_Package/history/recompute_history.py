#!/usr/bin/env python3
"""Read-only recount of archived Phase 2 judgments and current exemplar inputs.

No models, network, or repository imports. This audits the saved inputs at the
fixed commit; it does not reconstruct an unarchived historical run command.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshots', type=Path, default=Path(__file__).parent / 'snapshots')
    ap.add_argument('--output', type=Path, default=Path(__file__).parent / 'history_recomputed.json')
    args = ap.parse_args()
    def read(name):
        return json.loads((args.snapshots / name).read_text(encoding='utf-8'))
    judgment_names = [
        'phase2_v2_triggers_vs_baseline',
        'phase2_v16_vs_baseline_long',
        'phase2_v16_holdout50_vs_baseline_long_holdout50',
        'phase2_v16_holdout50_vs_phase2_v13_reflect_holdout50',
    ]
    judgments = {}
    for name in judgment_names:
        data = read('phase_two__analysis__cache__judgments__' + name + '.json')
        judgments[name] = {'n': len(data), 'winner_counts': dict(Counter(row.get('winner') for row in data.values()))}
    prefix = 'phase_two__analysis__cache__'
    sample = read(prefix + 'sample_100.json')
    heldout = read(prefix + 'sample_holdout_50.json')
    selections = read(prefix + 'phase2_v3_selections.json')
    exemplars = read(prefix + 'wisdom_diverse_exemplars.json')
    wisdom = {row['id']: row for row in read(prefix + 'wisdom_library.json')}
    ids = {row['problem_id'] for row in sample}
    heldout_ids = {row['problem_id'] for row in heldout}
    collisions = []
    case_count = 0
    for task in sample:
        pid, domain = task['problem_id'], task['domain']
        if domain in {'mathematics', 'science'}:
            continue
        case_count += 1
        # Exact default wisdom filtering and cross-domain fill from v16 lines 187–205.
        selected = [wisdom[sid] for sid in selections.get(pid, [])[:2] if sid in wisdom]
        selected = [w for w in selected if w['id'] in exemplars]
        if not selected:
            selected = [wisdom[sid] for sid in selections.get(pid, [])[:2] if sid in wisdom]
        for w in selected:
            pre_mined = exemplars.get(w['id'], [])
            cross = [e for e in pre_mined if e.get('domain') != domain][:3]
            if len(cross) < 3:
                cross += [e for e in pre_mined if e not in cross][:3-len(cross)]
            for e in cross:
                if e.get('pid') == pid:
                    collisions.append({'task': pid, 'wisdom': w['id']})
    exemplar_ids = {row.get('pid') for rows in exemplars.values() for row in rows if row.get('pid')}
    result = {
        'scope': 'Archived judgments plus current fixed-commit cache identities. Not a paid rerun or proof of historical run contamination.',
        'judgments': judgments,
        'sample100': len(sample), 'holdout50': len(heldout), 'id_overlap': sorted(ids & heldout_ids),
        'selection_key_count': len(selections),
        'holdout_with_default_selection': len(heldout_ids & set(selections)),
        'exemplar_entry_count': len(exemplars),
        'case_path_original_count': case_count,
        'tasks_with_own_prior_answer_exemplar': len({row['task'] for row in collisions}),
        'self_exemplar_instances': collisions,
        'all_exemplar_pids_count': len(exemplar_ids),
        'holdout_pids_in_exemplars': sorted(heldout_ids & exemplar_ids),
        'scar_denominators': {
            'typed_gate_removed_of_previously_selectable': 715 / 723,
            'full_abstention_of_primary_variants': 719 / 724,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'judgments': judgments, 'self_exemplar_tasks': result['tasks_with_own_prior_answer_exemplar']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

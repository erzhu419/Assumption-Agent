"""Study-only queue and question-local cost-history ablations."""
from copy import deepcopy


def keep_existing_pending(self, pending):
    return pending


def keep_question_without_new_cost(question, visit_costs):
    return deepcopy(question)


def install_study_ablation(arm):
    if arm=='prior_pair_queue':
        from hegel_machine.cognition.unified_session import UnifiedResearchSession
        UnifiedResearchSession._offer_pair_confirmations = keep_existing_pending
    elif arm=='no_pair_cost_feedback':
        from hegel_machine.cognition import pair_confirmation_revisits
        pair_confirmation_revisits.record_pair_revisit_cost = keep_question_without_new_cost

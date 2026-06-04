"""Cross-domain structural morphism benchmark.

This benchmark is deliberately not a broad semantic embedding benchmark.  It
isolates the claim that a bounded category-inspired morphism representation
can retrieve structural analogies that surface KG triples and lexical
embedding-style retrieval miss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "back",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "to",
    "while",
    "with",
}


@dataclass(frozen=True)
class MorphismSignature:
    signature_id: str
    label: str
    domain: str
    surface_text: str
    kg_triples: list[tuple[str, str, str]]
    objects: list[str] = field(default_factory=list)
    morphisms: list[str] = field(default_factory=list)
    composition_laws: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    negative_invariants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "signature_id": self.signature_id,
            "label": self.label,
            "domain": self.domain,
            "surface_text": self.surface_text,
            "kg_triples": [list(row) for row in self.kg_triples],
            "objects": list(self.objects),
            "morphisms": list(self.morphisms),
            "composition_laws": list(self.composition_laws),
            "invariants": list(self.invariants),
            "negative_invariants": list(self.negative_invariants),
        }


@dataclass(frozen=True)
class MorphismBenchmarkCase:
    case_id: str
    query: MorphismSignature
    expected_candidate_id: str
    candidates: list[MorphismSignature]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query.to_dict(),
            "expected_candidate_id": self.expected_candidate_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def build_morphism_independent_benchmark_payload(
    *,
    eval_id: str | None = None,
    neural_embedding_backend: str | None = "none",
    neural_embedding_model: str | None = None,
) -> dict:
    cases = _default_cases()
    neural_baseline = _build_neural_embedding_baseline(
        cases,
        backend=neural_embedding_backend or "none",
        model_name=neural_embedding_model,
    )
    rows = [
        _evaluate_case(
            case,
            neural_scores=neural_baseline.get("scores", {}).get(case.case_id),
        )
        for case in cases
    ]
    scorers = ("morphism", "kg_triple", "embedding_proxy") + (
        ("neural_embedding",) if neural_baseline.get("enabled") else ()
    )
    scorer_hit_rates = {
        scorer: round(sum(1 for row in rows if row["top_ids"][scorer] == row["expected_candidate_id"]) / len(rows), 4)
        for scorer in scorers
    }
    nonlexical_success_count = sum(1 for row in rows if row["nonlexical_structural_success"])
    nonlexical_success_rate = round(nonlexical_success_count / len(rows), 4) if rows else 0.0
    baseline_scorers = [scorer for scorer in scorers if scorer != "morphism"]
    best_baseline_rate = max(scorer_hit_rates[scorer] for scorer in baseline_scorers)
    morphism_margin = round(scorer_hit_rates["morphism"] - best_baseline_rate, 4)
    gates = [
        {
            "gate": "case_count",
            "pass": len(rows) >= 8,
            "observed": len(rows),
        },
        {
            "gate": "morphism_top1_rate",
            "pass": scorer_hit_rates["morphism"] >= 0.80,
            "observed": scorer_hit_rates["morphism"],
        },
        {
            "gate": "beats_surface_baselines",
            "pass": morphism_margin >= 0.20,
            "observed": {
                "morphism_rate": scorer_hit_rates["morphism"],
                "kg_triple_rate": scorer_hit_rates["kg_triple"],
                "embedding_proxy_rate": scorer_hit_rates["embedding_proxy"],
                "neural_embedding_rate": scorer_hit_rates.get("neural_embedding"),
                "margin": morphism_margin,
            },
        },
        {
            "gate": "nonlexical_success_rate",
            "pass": nonlexical_success_rate >= 0.75,
            "observed": nonlexical_success_rate,
        },
    ]
    if neural_baseline.get("requested"):
        gates.extend([
            {
                "gate": "neural_embedding_baseline_available",
                "pass": bool(neural_baseline.get("enabled")),
                "observed": {
                    "backend": neural_baseline.get("backend"),
                    "model": neural_baseline.get("model"),
                    "status": neural_baseline.get("status"),
                    "error": neural_baseline.get("error"),
                },
            },
            {
                "gate": "beats_neural_embedding_baseline",
                "pass": (
                    bool(neural_baseline.get("enabled"))
                    and scorer_hit_rates["morphism"] - scorer_hit_rates.get("neural_embedding", 1.0) >= 0.20
                ),
                "observed": {
                    "morphism_rate": scorer_hit_rates["morphism"],
                    "neural_embedding_rate": scorer_hit_rates.get("neural_embedding"),
                    "margin": round(
                        scorer_hit_rates["morphism"] - scorer_hit_rates.get("neural_embedding", 0.0),
                        4,
                    ) if neural_baseline.get("enabled") else None,
                },
            },
        ])
    return {
        "eval_id": eval_id or "morphism_independent_benchmark_20260604",
        "eval_kind": "morphism_independent_cross_domain_benchmark",
        "case_count": len(rows),
        "scorer_hit_rates": scorer_hit_rates,
        "morphism_margin_over_best_baseline": morphism_margin,
        "nonlexical_success_rate": nonlexical_success_rate,
        "neural_embedding_baseline": {
            key: value
            for key, value in neural_baseline.items()
            if key != "scores"
        },
        "gates": gates,
        "pass": all(gate["pass"] for gate in gates),
        "baseline_note": (
            "embedding_proxy is deterministic lexical cosine over task text.  "
            "neural_embedding, when enabled, is a real sentence-embedding retrieval baseline over "
            "the same surface text."
        ),
        "rows": rows,
    }


def _evaluate_case(case: MorphismBenchmarkCase, *, neural_scores: dict[str, float] | None = None) -> dict:
    candidate_rows = []
    for candidate in case.candidates:
        morphism = _morphism_score(case.query, candidate)
        kg = _kg_triple_score(case.query, candidate)
        embedding = _embedding_proxy_score(case.query, candidate)
        scores = {
            "morphism": morphism,
            "kg_triple": kg,
            "embedding_proxy": embedding,
        }
        if neural_scores is not None:
            scores["neural_embedding"] = round(float(neural_scores.get(candidate.signature_id, 0.0)), 4)
        candidate_rows.append({
            "candidate_id": candidate.signature_id,
            "label": candidate.label,
            "domain": candidate.domain,
            "scores": scores,
            "morphism_evidence": _morphism_evidence(case.query, candidate),
        })
    scorers = tuple(candidate_rows[0]["scores"]) if candidate_rows else ()
    ranks = {
        scorer: sorted(candidate_rows, key=lambda row: (-row["scores"][scorer], row["candidate_id"]))
        for scorer in scorers
    }
    top_ids = {scorer: ranked[0]["candidate_id"] for scorer, ranked in ranks.items()}
    expected = next(row for row in candidate_rows if row["candidate_id"] == case.expected_candidate_id)
    false_rows = [row for row in candidate_rows if row["candidate_id"] != case.expected_candidate_id]
    baseline_scorers = [scorer for scorer in scorers if scorer != "morphism"]
    best_false_by_baseline = {
        scorer: max(row["scores"][scorer] for row in false_rows)
        for scorer in baseline_scorers
    }
    nonlexical_structural_success = (
        top_ids["morphism"] == case.expected_candidate_id
        and all(expected["scores"][scorer] < best_false_by_baseline[scorer] for scorer in baseline_scorers)
    )
    return {
        "case_id": case.case_id,
        "query_label": case.query.label,
        "query_domain": case.query.domain,
        "expected_candidate_id": case.expected_candidate_id,
        "top_ids": top_ids,
        "passed_by": {
            scorer: top_ids[scorer] == case.expected_candidate_id
            for scorer in scorers
        },
        "nonlexical_structural_success": nonlexical_structural_success,
        "candidate_scores": candidate_rows,
        "rankings": {
            scorer: [
                {"candidate_id": row["candidate_id"], "score": row["scores"][scorer]}
                for row in ranked
            ]
            for scorer, ranked in ranks.items()
        },
    }


def _build_neural_embedding_baseline(
    cases: list[MorphismBenchmarkCase],
    *,
    backend: str,
    model_name: str | None,
) -> dict:
    backend = (backend or "none").strip().lower()
    if backend in {"none", "off", "false", "0"}:
        return {
            "requested": False,
            "enabled": False,
            "backend": "none",
            "model": None,
            "status": "disabled",
        }
    if backend == "auto":
        try:
            return _sentence_transformer_baseline(cases, model_name=model_name)
        except Exception as exc:
            try:
                return _openai_embedding_baseline(cases, model_name=model_name)
            except Exception as openai_exc:
                return {
                    "requested": True,
                    "enabled": False,
                    "backend": "auto",
                    "model": model_name,
                    "status": "unavailable",
                    "error": f"sentence_transformer={exc}; openai={openai_exc}",
                }
    if backend in {"sentence_transformer", "sentence-transformer", "sbert", "minilm"}:
        return _sentence_transformer_baseline(cases, model_name=model_name)
    if backend in {"openai", "newapi", "api"}:
        return _openai_embedding_baseline(cases, model_name=model_name)
    raise ValueError(f"Unknown neural embedding backend: {backend}")


def _sentence_transformer_baseline(cases: list[MorphismBenchmarkCase], *, model_name: str | None) -> dict:
    model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence_transformers is not installed") from exc

    texts = _embedding_texts(cases)
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:
        raise RuntimeError(f"sentence_transformer model unavailable: {model_name}") from exc
    scores = _embedding_scores_from_vectors(cases, texts, embeddings)
    return {
        "requested": True,
        "enabled": True,
        "backend": "sentence_transformer",
        "model": model_name,
        "status": "ok",
        "text_field": "surface_text",
        "scores": scores,
    }


def _openai_embedding_baseline(cases: list[MorphismBenchmarkCase], *, model_name: str | None) -> dict:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai SDK is not installed") from exc

    api_key = os.environ.get("ASSUMPTION_OS_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSUMPTION_OS_EMBEDDING_API_KEY or OPENAI_API_KEY is not set")
    base_url = os.environ.get("ASSUMPTION_OS_EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model_name = model_name or os.environ.get("ASSUMPTION_OS_EMBEDDING_MODEL") or "text-embedding-3-small"
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    texts = _embedding_texts(cases)
    response = client.embeddings.create(model=model_name, input=texts)
    embeddings = [row.embedding for row in response.data]
    scores = _embedding_scores_from_vectors(cases, texts, embeddings)
    return {
        "requested": True,
        "enabled": True,
        "backend": "openai_compatible",
        "model": model_name,
        "base_url_configured": bool(base_url),
        "status": "ok",
        "text_field": "surface_text",
        "scores": scores,
    }


def _embedding_texts(cases: list[MorphismBenchmarkCase]) -> list[str]:
    texts = []
    seen = set()
    for case in cases:
        for signature in [case.query, *case.candidates]:
            if signature.surface_text not in seen:
                seen.add(signature.surface_text)
                texts.append(signature.surface_text)
    return texts


def _embedding_scores_from_vectors(
    cases: list[MorphismBenchmarkCase],
    texts: list[str],
    embeddings,
) -> dict[str, dict[str, float]]:
    vector_by_text = {text: embeddings[idx] for idx, text in enumerate(texts)}
    scores: dict[str, dict[str, float]] = {}
    for case in cases:
        query_vector = vector_by_text[case.query.surface_text]
        scores[case.case_id] = {
            candidate.signature_id: round(_vector_cosine(query_vector, vector_by_text[candidate.surface_text]), 4)
            for candidate in case.candidates
        }
    return scores


def _vector_cosine(left, right) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        fa = float(a)
        fb = float(b)
        dot += fa * fb
        left_norm += fa * fa
        right_norm += fb * fb
    if not left_norm or not right_norm:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _morphism_score(query: MorphismSignature, candidate: MorphismSignature) -> float:
    score = (
        0.18 * _jaccard(query.objects, candidate.objects)
        + 0.30 * _jaccard(query.morphisms, candidate.morphisms)
        + 0.22 * _jaccard(query.composition_laws, candidate.composition_laws)
        + 0.30 * _jaccard(query.invariants, candidate.invariants)
    )
    broken_penalty = 0.12 * _jaccard(query.invariants, candidate.negative_invariants)
    return round(max(0.0, score - broken_penalty), 4)


def _kg_triple_score(query: MorphismSignature, candidate: MorphismSignature) -> float:
    query_triple_text = _triple_text(query.kg_triples)
    candidate_triple_text = _triple_text(candidate.kg_triples)
    token_score = _counter_cosine(_tokens(query_triple_text), _tokens(candidate_triple_text))
    predicate_score = _jaccard(
        [pred for _, pred, _ in query.kg_triples],
        [pred for _, pred, _ in candidate.kg_triples],
    )
    entity_score = _jaccard(
        [part for subj, _, obj in query.kg_triples for part in (subj, obj)],
        [part for subj, _, obj in candidate.kg_triples for part in (subj, obj)],
    )
    return round(0.65 * token_score + 0.20 * predicate_score + 0.15 * entity_score, 4)


def _embedding_proxy_score(query: MorphismSignature, candidate: MorphismSignature) -> float:
    return round(_counter_cosine(_tokens(query.surface_text), _tokens(candidate.surface_text)), 4)


def _morphism_evidence(query: MorphismSignature, candidate: MorphismSignature) -> dict:
    return {
        "object_overlap": sorted(set(query.objects) & set(candidate.objects)),
        "morphism_overlap": sorted(set(query.morphisms) & set(candidate.morphisms)),
        "composition_overlap": sorted(set(query.composition_laws) & set(candidate.composition_laws)),
        "invariant_overlap": sorted(set(query.invariants) & set(candidate.invariants)),
        "broken_query_invariants": sorted(set(query.invariants) & set(candidate.negative_invariants)),
    }


def _tokens(text: str) -> Counter:
    terms = [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]
    return Counter(terms)


def _triple_text(triples: list[tuple[str, str, str]]) -> str:
    return " ".join(" ".join(row) for row in triples)


def _counter_cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right[token] for token in set(left) & set(right))
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = {item.lower().strip() for item in left if item}
    right_set = {item.lower().strip() for item in right if item}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _sig(
    signature_id: str,
    label: str,
    domain: str,
    surface_text: str,
    kg_triples: list[tuple[str, str, str]],
    objects: list[str],
    morphisms: list[str],
    composition_laws: list[str],
    invariants: list[str],
    negative_invariants: list[str] | None = None,
) -> MorphismSignature:
    return MorphismSignature(
        signature_id=signature_id,
        label=label,
        domain=domain,
        surface_text=surface_text,
        kg_triples=kg_triples,
        objects=objects,
        morphisms=morphisms,
        composition_laws=composition_laws,
        invariants=invariants,
        negative_invariants=negative_invariants or [],
    )


def _default_cases() -> list[MorphismBenchmarkCase]:
    return [
        MorphismBenchmarkCase(
            case_id="morph_le_chatelier_lenz",
            query=_sig(
                "q_le_chatelier",
                "Le Chatelier response",
                "chemistry",
                "Chemical equilibrium receives concentration or temperature stress; the reaction shifts direction to counter the stress and restore equilibrium.",
                [
                    ("chemical equilibrium", "is disturbed by", "temperature stress"),
                    ("reaction", "shifts to counter", "chemical stress"),
                ],
                ["system_state", "external_perturbation", "opposing_response", "regulated_quantity"],
                ["perturb", "measure_deviation", "oppose_change", "restore_constraint"],
                ["perturb_then_response_opposes_delta"],
                ["opposing_response_preserves_constraint", "response_direction_depends_on_disturbance_sign"],
            ),
            expected_candidate_id="c_lenz_law",
            candidates=[
                _sig(
                    "c_chemical_catalyst",
                    "Catalyst changes reaction speed",
                    "chemistry",
                    "A chemical catalyst speeds an equilibrium reaction and changes reaction rate without being consumed.",
                    [
                        ("chemical catalyst", "speeds", "chemical reaction"),
                        ("reaction", "has", "equilibrium rate"),
                    ],
                    ["system_state", "rate_modifier"],
                    ["accelerate_process"],
                    ["modifier_then_rate_changes"],
                    ["rate_changes_without_directional_restoration"],
                    ["opposing_response_preserves_constraint"],
                ),
                _sig(
                    "c_lenz_law",
                    "Lenz law induction",
                    "electromagnetism",
                    "Changing magnetic flux induces a current whose magnetic field acts against the flux change that caused it.",
                    [
                        ("magnetic flux", "induces", "electric current"),
                        ("current field", "acts against", "flux change"),
                    ],
                    ["system_state", "external_perturbation", "opposing_response", "regulated_quantity"],
                    ["perturb", "measure_deviation", "oppose_change", "restore_constraint"],
                    ["perturb_then_response_opposes_delta"],
                    ["opposing_response_preserves_constraint", "response_direction_depends_on_disturbance_sign"],
                ),
                _sig(
                    "c_temperature_rate_rule",
                    "Temperature rate rule",
                    "chemistry",
                    "Higher temperature often increases chemical reaction rate and changes equilibrium constants.",
                    [
                        ("temperature", "increases", "chemical reaction rate"),
                        ("equilibrium constant", "changes with", "temperature"),
                    ],
                    ["input_setting", "rate"],
                    ["increase_rate"],
                    ["input_then_rate_increases"],
                    ["monotone_input_response"],
                    ["opposing_response_preserves_constraint"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_resnet_kalman",
            query=_sig(
                "q_residual_block",
                "Residual block",
                "deep_learning",
                "A deep residual block carries identity activation around a stack and learns a residual correction added back at the output.",
                [
                    ("residual block", "carries", "identity activation"),
                    ("deep stack", "learns", "residual correction"),
                ],
                ["baseline_path", "delta_update", "join_node", "trainable_transform"],
                ["preserve_identity", "compute_delta", "add_delta", "fallback_when_delta_zero"],
                ["identity_plus_delta_composes_to_output"],
                ["zero_delta_preserves_input", "local_update_does_not_destroy_baseline"],
            ),
            expected_candidate_id="c_kalman_innovation",
            candidates=[
                _sig(
                    "c_deep_layer_scaling",
                    "Deep layer scaling",
                    "deep_learning",
                    "A deep convolutional network adds more layers, residual blocks, dropout, and learning-rate tuning.",
                    [
                        ("deep network", "adds", "more residual layers"),
                        ("optimizer", "tunes", "learning rate"),
                    ],
                    ["stack_depth", "optimizer_setting"],
                    ["increase_depth", "regularize"],
                    ["more_layers_then_capacity_changes"],
                    ["capacity_changes_with_depth"],
                    ["zero_delta_preserves_input"],
                ),
                _sig(
                    "c_kalman_innovation",
                    "Kalman innovation update",
                    "control_estimation",
                    "A state estimator carries a prior prediction and adds an innovation correction from the new measurement residual.",
                    [
                        ("state estimator", "predicts", "prior state"),
                        ("measurement residual", "updates", "posterior state"),
                    ],
                    ["baseline_path", "delta_update", "join_node", "trainable_transform"],
                    ["preserve_identity", "compute_delta", "add_delta", "fallback_when_delta_zero"],
                    ["identity_plus_delta_composes_to_output"],
                    ["zero_delta_preserves_input", "local_update_does_not_destroy_baseline"],
                ),
                _sig(
                    "c_skip_regularization",
                    "Skip regularization trick",
                    "deep_learning",
                    "Skip connections, normalization, and dropout make very deep nets easier to optimize.",
                    [
                        ("skip connection", "helps", "deep net optimization"),
                        ("dropout", "regularizes", "activation"),
                    ],
                    ["optimization_trick", "regularizer"],
                    ["stabilize_training"],
                    ["regularizer_then_training_smoother"],
                    ["optimization_stability"],
                    ["local_update_does_not_destroy_baseline"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_jepa_seismic",
            query=_sig(
                "q_jepa_noise",
                "JEPA latent nuisance separation",
                "world_model",
                "A JEPA-style model predicts latent world representations and ignores stochastic pixel noise, separating stable state from nuisance variation.",
                [
                    ("JEPA model", "predicts", "latent representation"),
                    ("pixel noise", "is ignored by", "world model"),
                ],
                ["stable_signal", "nuisance_noise", "latent_state", "predictor"],
                ["separate_signal", "suppress_noise", "validate_stable_feature"],
                ["suppress_noise_then_preserve_predictive_state"],
                ["predictable_structure_separated", "stochastic_nuisance_suppressed"],
            ),
            expected_candidate_id="c_seismic_autocorrelation",
            candidates=[
                _sig(
                    "c_autoencoder_pixels",
                    "Pixel autoencoder reconstruction",
                    "world_model",
                    "An image autoencoder reconstructs every pixel, texture, and stochastic detail in the training image.",
                    [
                        ("image autoencoder", "reconstructs", "pixel detail"),
                        ("pixel noise", "appears in", "reconstruction"),
                    ],
                    ["input_detail", "decoder"],
                    ["reconstruct_all_detail"],
                    ["detail_then_reconstruction"],
                    ["detail_preserved"],
                    ["stochastic_nuisance_suppressed"],
                ),
                _sig(
                    "c_seismic_autocorrelation",
                    "Seismic autocorrelation denoising",
                    "geophysics",
                    "Repeated seismic wave arrivals are stacked so random noise cancels while coherent subsurface reflections remain.",
                    [
                        ("seismic wave arrivals", "are stacked into", "coherent reflection"),
                        ("random noise", "cancels during", "autocorrelation stack"),
                    ],
                    ["stable_signal", "nuisance_noise", "latent_state", "predictor"],
                    ["separate_signal", "suppress_noise", "validate_stable_feature"],
                    ["suppress_noise_then_preserve_predictive_state"],
                    ["predictable_structure_separated", "stochastic_nuisance_suppressed"],
                ),
                _sig(
                    "c_latent_diffusion_prompt",
                    "Latent diffusion prompt detail",
                    "world_model",
                    "A latent diffusion prompt asks a model to generate rich pixel textures and stochastic image detail.",
                    [
                        ("latent model", "generates", "pixel texture"),
                        ("prompt", "controls", "image detail"),
                    ],
                    ["generator", "texture_detail"],
                    ["sample_detail"],
                    ["prompt_then_texture_changes"],
                    ["stochastic_detail_preserved"],
                    ["predictable_structure_separated"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_thermostat_stabilizer",
            query=_sig(
                "q_thermostat_feedback",
                "Thermostat negative feedback",
                "control",
                "A thermostat senses temperature deviation and drives heating or cooling opposite to the deviation until the setpoint is restored.",
                [
                    ("thermostat", "senses", "temperature deviation"),
                    ("controller", "drives opposite", "setpoint error"),
                ],
                ["setpoint", "measured_state", "error_signal", "opposing_actuator"],
                ["sense_error", "choose_opposing_action", "reduce_error"],
                ["error_measure_then_opposing_actuation_then_error_reduces"],
                ["negative_feedback_reduces_deviation", "closed_loop_uses_measured_error"],
            ),
            expected_candidate_id="c_automatic_stabilizer",
            candidates=[
                _sig(
                    "c_smart_thermostat_schedule",
                    "Smart thermostat schedule",
                    "control",
                    "A smart thermostat learns a temperature schedule and user comfort preferences for heating and cooling.",
                    [
                        ("smart thermostat", "learns", "temperature schedule"),
                        ("user preference", "sets", "comfort target"),
                    ],
                    ["schedule", "preference"],
                    ["learn_preference"],
                    ["schedule_then_temperature_changes"],
                    ["preference_following"],
                    ["negative_feedback_reduces_deviation"],
                ),
                _sig(
                    "c_automatic_stabilizer",
                    "Economic automatic stabilizer",
                    "macro_economics",
                    "Unemployment benefits and progressive taxes move countercyclically, dampening demand shocks without discretionary intervention.",
                    [
                        ("unemployment benefits", "rise during", "downturn"),
                        ("tax intake", "falls during", "negative output gap"),
                    ],
                    ["setpoint", "measured_state", "error_signal", "opposing_actuator"],
                    ["sense_error", "choose_opposing_action", "reduce_error"],
                    ["error_measure_then_opposing_actuation_then_error_reduces"],
                    ["negative_feedback_reduces_deviation", "closed_loop_uses_measured_error"],
                ),
                _sig(
                    "c_heater_power_upgrade",
                    "Heater power upgrade",
                    "control",
                    "A larger heater increases temperature faster and improves comfort in a cold room.",
                    [
                        ("heater", "increases", "temperature"),
                        ("power upgrade", "improves", "comfort"),
                    ],
                    ["actuator_capacity", "output_power"],
                    ["increase_actuation"],
                    ["power_then_temperature_rises"],
                    ["monotone_actuator_response"],
                    ["closed_loop_uses_measured_error"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_ci_enzyme_bottleneck",
            query=_sig(
                "q_ci_bottleneck",
                "CI pipeline bottleneck",
                "software_engineering",
                "A release pipeline queues at the slow signing step; speeding unrelated packaging stages does not raise end-to-end throughput.",
                [
                    ("release pipeline", "queues at", "signing step"),
                    ("packaging stage", "does not raise", "end throughput"),
                ],
                ["flow_item", "capacity_limiter", "upstream_stage", "downstream_output"],
                ["accumulate_queue", "saturate_limiter", "propagate_throughput_bound"],
                ["upstream_speedup_then_no_output_gain_if_limiter_unchanged"],
                ["bottleneck_controls_throughput", "capacity_constraint_explicit"],
            ),
            expected_candidate_id="c_enzyme_saturation",
            candidates=[
                _sig(
                    "c_more_ci_parallelism",
                    "More CI parallelism",
                    "software_engineering",
                    "The release pipeline adds parallel packaging jobs, faster tests, and more build workers.",
                    [
                        ("release pipeline", "adds", "parallel packaging"),
                        ("build workers", "speed", "CI tests"),
                    ],
                    ["parallel_worker", "upstream_stage"],
                    ["parallelize_stage"],
                    ["workers_then_stage_faster"],
                    ["local_stage_speed_changes"],
                    ["bottleneck_controls_throughput"],
                ),
                _sig(
                    "c_enzyme_saturation",
                    "Enzyme saturation limit",
                    "biochemistry",
                    "Product flux is capped by a slow active-site step; adding more upstream substrate after saturation does not increase reaction output.",
                    [
                        ("active site", "limits", "product flux"),
                        ("substrate", "does not raise", "saturated reaction output"),
                    ],
                    ["flow_item", "capacity_limiter", "upstream_stage", "downstream_output"],
                    ["accumulate_queue", "saturate_limiter", "propagate_throughput_bound"],
                    ["upstream_speedup_then_no_output_gain_if_limiter_unchanged"],
                    ["bottleneck_controls_throughput", "capacity_constraint_explicit"],
                ),
                _sig(
                    "c_ci_dashboard",
                    "CI dashboard visibility",
                    "software_engineering",
                    "A CI dashboard displays release pipeline steps, queue time, signing status, and packaging logs.",
                    [
                        ("CI dashboard", "displays", "release pipeline"),
                        ("signing status", "appears in", "packaging logs"),
                    ],
                    ["observer", "status_view"],
                    ["display_state"],
                    ["observe_then_report"],
                    ["observability_improves"],
                    ["capacity_constraint_explicit"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_strangler_bridge",
            query=_sig(
                "q_strangler_migration",
                "Strangler migration",
                "software_engineering",
                "A legacy service is wrapped, a small traffic slice goes to the new module, and rollback keeps the old path available.",
                [
                    ("legacy service", "is wrapped by", "routing facade"),
                    ("traffic slice", "goes to", "new module"),
                ],
                ["old_path", "new_component", "boundary_wrapper", "rollback_path"],
                ["route_slice", "verify_slice", "expand_or_rollback"],
                ["slice_then_verify_then_expand_preserves_service"],
                ["incremental_replacement_preserves_interface", "rollback_path_remains_available"],
            ),
            expected_candidate_id="c_bridge_retrofit",
            candidates=[
                _sig(
                    "c_big_bang_rewrite",
                    "Big-bang monolith rewrite",
                    "software_engineering",
                    "The legacy service rewrite replaces the whole monolith in one launch with a new module and no parallel old path.",
                    [
                        ("legacy service", "is replaced by", "new module"),
                        ("monolith", "launches as", "one rewrite"),
                    ],
                    ["old_system", "new_system"],
                    ["replace_all"],
                    ["replace_all_then_cutover"],
                    ["single_cutover"],
                    ["rollback_path_remains_available"],
                ),
                _sig(
                    "c_bridge_retrofit",
                    "Bridge retrofit under traffic",
                    "civil_engineering",
                    "Temporary support carries loads while one bridge segment is replaced, inspected, and opened before the next segment is touched.",
                    [
                        ("temporary support", "carries", "bridge load"),
                        ("bridge segment", "is replaced after", "inspection"),
                    ],
                    ["old_path", "new_component", "boundary_wrapper", "rollback_path"],
                    ["route_slice", "verify_slice", "expand_or_rollback"],
                    ["slice_then_verify_then_expand_preserves_service"],
                    ["incremental_replacement_preserves_interface", "rollback_path_remains_available"],
                ),
                _sig(
                    "c_legacy_inventory",
                    "Legacy inventory catalog",
                    "software_engineering",
                    "A legacy service inventory lists old modules, endpoints, traffic, and owners.",
                    [
                        ("legacy service", "has", "module inventory"),
                        ("endpoint", "has", "traffic owner"),
                    ],
                    ["catalog", "owner"],
                    ["inventory"],
                    ["catalog_then_prioritize"],
                    ["visibility_precedes_planning"],
                    ["incremental_replacement_preserves_interface"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_abtest_clinical_trial",
            query=_sig(
                "q_checkout_abtest",
                "Checkout A/B intervention",
                "product_experiment",
                "An A/B test changes one checkout variable while matched control traffic holds time, audience, and offer constant.",
                [
                    ("A/B test", "changes", "checkout variable"),
                    ("control traffic", "holds constant", "audience and time"),
                ],
                ["intervention_row", "control_row", "matched_context", "measured_outcome"],
                ["change_one_factor", "hold_controls", "compare_outcomes"],
                ["single_intervention_then_matched_comparison"],
                ["single_intervention_isolated", "matched_control_required"],
            ),
            expected_candidate_id="c_randomized_clinical_trial",
            candidates=[
                _sig(
                    "c_checkout_segmentation",
                    "Checkout segmentation dashboard",
                    "product_experiment",
                    "A checkout dashboard segments A/B traffic by offer, audience, time, browser, and many product variables.",
                    [
                        ("checkout dashboard", "segments", "A/B traffic"),
                        ("product variables", "vary by", "audience and time"),
                    ],
                    ["segment", "variable_set"],
                    ["slice_many_factors"],
                    ["many_slices_then_dashboard"],
                    ["observational_segmentation"],
                    ["single_intervention_isolated"],
                ),
                _sig(
                    "c_randomized_clinical_trial",
                    "Randomized clinical trial",
                    "medicine",
                    "A trial gives one treatment to a randomized arm while a matched control arm receives standard care and outcomes are compared.",
                    [
                        ("randomized trial", "assigns", "treatment arm"),
                        ("control arm", "receives", "standard care"),
                    ],
                    ["intervention_row", "control_row", "matched_context", "measured_outcome"],
                    ["change_one_factor", "hold_controls", "compare_outcomes"],
                    ["single_intervention_then_matched_comparison"],
                    ["single_intervention_isolated", "matched_control_required"],
                ),
                _sig(
                    "c_checkout_copy_refresh",
                    "Checkout copy refresh",
                    "product_experiment",
                    "Checkout wording, offer copy, audience targeting, and timing are all refreshed before launch.",
                    [
                        ("checkout copy", "changes", "offer wording"),
                        ("audience targeting", "changes with", "launch timing"),
                    ],
                    ["multi_change_release"],
                    ["change_many_factors"],
                    ["many_changes_then_launch"],
                    ["confounded_release"],
                    ["matched_control_required"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_counterexample_flight_envelope",
            query=_sig(
                "q_boundary_counterexample",
                "Boundary counterexample refinement",
                "mathematics",
                "A theorem candidate fails on a boundary case, so the claim is weakened and a guard condition is added.",
                [
                    ("theorem candidate", "fails on", "boundary case"),
                    ("claim", "is weakened by", "guard condition"),
                ],
                ["general_claim", "counterexample", "refined_claim", "validity_region"],
                ["falsify_claim", "localize_failure", "narrow_scope"],
                ["counterexample_then_scope_refinement"],
                ["counterexample_refines_claim", "revised_scope_blocks_same_failure"],
            ),
            expected_candidate_id="c_flight_envelope_protection",
            candidates=[
                _sig(
                    "c_elegant_theorem_proof",
                    "Elegant theorem proof",
                    "mathematics",
                    "A theorem proof explains a beautiful boundary case and derives a concise guard condition.",
                    [
                        ("theorem proof", "explains", "boundary case"),
                        ("guard condition", "appears in", "mathematical claim"),
                    ],
                    ["proof_text", "claim"],
                    ["explain_case"],
                    ["proof_then_claim"],
                    ["exposition_quality"],
                    ["counterexample_refines_claim"],
                ),
                _sig(
                    "c_flight_envelope_protection",
                    "Flight envelope protection",
                    "aerospace_control",
                    "A stall edge case breaks the control law, so the controller narrows the safe envelope and adds protection logic.",
                    [
                        ("stall edge case", "breaks", "control law"),
                        ("flight envelope", "is narrowed by", "protection logic"),
                    ],
                    ["general_claim", "counterexample", "refined_claim", "validity_region"],
                    ["falsify_claim", "localize_failure", "narrow_scope"],
                    ["counterexample_then_scope_refinement"],
                    ["counterexample_refines_claim", "revised_scope_blocks_same_failure"],
                ),
                _sig(
                    "c_boundary_value_problem",
                    "Boundary value equation",
                    "mathematics",
                    "A boundary value problem solves a differential equation with boundary conditions.",
                    [
                        ("boundary value problem", "solves", "differential equation"),
                        ("boundary condition", "sets", "solution value"),
                    ],
                    ["equation", "boundary_condition"],
                    ["solve_equation"],
                    ["condition_then_solution"],
                    ["well_posed_boundary"],
                    ["revised_scope_blocks_same_failure"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_budget_mass_balance",
            query=_sig(
                "q_budget_balance",
                "Budget balance transfer",
                "finance",
                "A budget transfer must close before-after accounting: money added to one line is removed from another line.",
                [
                    ("budget transfer", "moves", "money"),
                    ("accounting balance", "closes", "before after ledger"),
                ],
                ["conserved_quantity", "source_bucket", "target_bucket", "balance_check"],
                ["remove_from_source", "add_to_target", "verify_balance"],
                ["remove_then_add_then_balance_closes"],
                ["conserved_quantity_preserved", "balance_check_closes"],
            ),
            expected_candidate_id="c_reactor_mass_balance",
            candidates=[
                _sig(
                    "c_budget_dashboard",
                    "Budget category dashboard",
                    "finance",
                    "A budget app shows transfers, money categories, monthly lines, and accounting charts.",
                    [
                        ("budget app", "shows", "money categories"),
                        ("accounting chart", "displays", "monthly lines"),
                    ],
                    ["dashboard", "category"],
                    ["display_balance"],
                    ["display_then_review"],
                    ["visibility_of_accounts"],
                    ["conserved_quantity_preserved"],
                ),
                _sig(
                    "c_reactor_mass_balance",
                    "Reactor mass balance",
                    "chemical_engineering",
                    "A reactor balance tracks inflow, outflow, and accumulation so material created in one term is removed from another term.",
                    [
                        ("reactor balance", "tracks", "inflow and outflow"),
                        ("accumulation term", "closes", "mass ledger"),
                    ],
                    ["conserved_quantity", "source_bucket", "target_bucket", "balance_check"],
                    ["remove_from_source", "add_to_target", "verify_balance"],
                    ["remove_then_add_then_balance_closes"],
                    ["conserved_quantity_preserved", "balance_check_closes"],
                ),
                _sig(
                    "c_budget_growth_plan",
                    "Budget growth plan",
                    "finance",
                    "A budget plan increases one spending line after revenue growth and changes several accounting assumptions.",
                    [
                        ("budget plan", "increases", "spending line"),
                        ("revenue growth", "changes", "accounting assumption"),
                    ],
                    ["growth_input", "spending_line"],
                    ["increase_allocation"],
                    ["growth_then_budget_expands"],
                    ["new_money_enters_system"],
                    ["balance_check_closes"],
                ),
            ],
        ),
        MorphismBenchmarkCase(
            case_id="morph_compiler_assembly",
            query=_sig(
                "q_compiler_composition",
                "Compiler module composition",
                "software_engineering",
                "A compiler splits a program into independent modules with interface contracts, then linking verifies whole executable behavior.",
                [
                    ("compiler", "splits", "program modules"),
                    ("linker", "verifies", "whole executable"),
                ],
                ["root_problem", "subproblem", "interface_contract", "composition_check"],
                ["decompose", "solve_subproblem", "compose_solution"],
                ["subsolutions_then_interface_join_then_parent_goal"],
                ["interfaces_preserve_parent_goal", "composition_check_required"],
            ),
            expected_candidate_id="c_manufacturing_subassembly",
            candidates=[
                _sig(
                    "c_compiler_optimizer",
                    "Compiler optimizer pass",
                    "software_engineering",
                    "A compiler optimizer renames variables, rewrites modules, and formats intermediate code before linking.",
                    [
                        ("compiler optimizer", "rewrites", "program modules"),
                        ("linker", "uses", "intermediate code"),
                    ],
                    ["optimizer", "code_unit"],
                    ["rewrite_code"],
                    ["rewrite_then_optimize"],
                    ["local_optimization"],
                    ["composition_check_required"],
                ),
                _sig(
                    "c_manufacturing_subassembly",
                    "Manufacturing subassembly fit",
                    "manufacturing",
                    "A product is split into subassemblies with mating tolerances; final assembly checks that the whole unit fits and works.",
                    [
                        ("manufacturing cell", "builds", "subassembly"),
                        ("final assembly", "verifies", "whole product"),
                    ],
                    ["root_problem", "subproblem", "interface_contract", "composition_check"],
                    ["decompose", "solve_subproblem", "compose_solution"],
                    ["subsolutions_then_interface_join_then_parent_goal"],
                    ["interfaces_preserve_parent_goal", "composition_check_required"],
                ),
                _sig(
                    "c_module_inventory",
                    "Module inventory",
                    "software_engineering",
                    "A program module inventory lists dependencies, interfaces, link order, and executable owners.",
                    [
                        ("module inventory", "lists", "interfaces"),
                        ("link order", "belongs to", "executable owner"),
                    ],
                    ["inventory", "dependency"],
                    ["catalog_modules"],
                    ["catalog_then_plan"],
                    ["visibility_precedes_composition"],
                    ["interfaces_preserve_parent_goal"],
                ),
            ],
        ),
    ]


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-id", default="morphism_independent_benchmark_20260604")
    parser.add_argument(
        "--neural-embedding-backend",
        default="none",
        choices=["none", "auto", "sentence_transformer", "openai"],
    )
    parser.add_argument("--neural-embedding-model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    payload = build_morphism_independent_benchmark_payload(
        eval_id=args.eval_id,
        neural_embedding_backend=args.neural_embedding_backend,
        neural_embedding_model=args.neural_embedding_model,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    _main()

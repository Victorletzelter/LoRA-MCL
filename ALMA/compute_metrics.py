import pickle
import argparse
import os
import sys
import torch
import yaml
import numpy as np
from itertools import combinations
from sacrebleu.metrics import BLEU
from sacrebleu import sentence_bleu
from typing import List, Dict, Any, Tuple, Optional
import traceback

import logging
pylog = logging.getLogger(__name__)

from typing import List, Dict, Any, Tuple, Iterable
from sacrebleu.metrics import BLEU
from sacrebleu import corpus_bleu as _corpus_bleu

def detok_moses(line: str, lang: str = "de") -> str:
    # If your text is *Moses-tokenized* (e.g., "das ist !"), detok with sacremoses
    # pip install sacremoses
    try:
        from sacremoses import MosesDetokenizer
        md = MosesDetokenizer(lang=lang)
        # MosesDetokenizer expects *tokenized* input: split on spaces
        return md.detokenize(line.split())
    except Exception:
        # fall back gracefully if sacremoses isn't available
        return line

def detok_lines(lines: Iterable[str], scheme: str = "none", lang: str = "de") -> list:
    if scheme == "none":
        return list(lines)
    # if scheme == "bpe":
    #     return [detok_bpe(x) for x in lines]
    # if scheme == "spm":
    #     return [detok_spm(x) for x in lines]
    if scheme == "moses":
        return [detok_moses(x, lang=lang) for x in lines]
    raise ValueError(f"Unknown detokenization scheme: {scheme}")

def detok_nested_refs(refs: list[list[str]], scheme: str = "none", lang: str = "de") -> list[list[str]]:
    # refs is [[ref1, ref2, ...], ...]
    return [detok_lines(r, scheme=scheme, lang=lang) for r in refs]

###### Fairseq-style metrics ######

def corpus_bleu_sacre(sys_stream: List[str], ref_streams: List[List[str]], tokenizer: str = "none") -> float:
    """
    Corpus BLEU via SacreBLEU with explicit tokenizer.
    'ref_streams' must be a list of reference lists (one list per reference).
    """
    bleu = _corpus_bleu(sys_stream, ref_streams, tokenize=tokenizer)
    return bleu.score

def pairwise_bleu_fairseq_exact(
    all_predictions: Dict[int, List[str]], tokenizer: str = "none"
) -> float:
    """
    Fairseq-style Pairwise-BLEU.
    Treat each example's set of hypotheses as S = [h0, h1, ..., h_{K-1}].
    Build two flat lists:
      _ref += S[i] for all i != j
      _hypo += S[j] for all i != j
    Then compute ONE corpus BLEU over (_hypo vs _ref).
    Returns the single Pairwise-BLEU number.
    """
    # all_predictions: {hyp_idx: [sent_0, sent_1, ...]}
    # First pivot to per-example: List[List[str]], each inner list is hypotheses for that example
    num_hypotheses = len(all_predictions)
    num_examples = len(next(iter(all_predictions.values())))
    per_example_hypos: List[List[str]] = []
    for ex in range(num_examples):
        per_example_hypos.append([all_predictions[h][ex] for h in range(num_hypotheses)])

    refs_flat, hypos_flat = [], []
    for S in per_example_hypos:
        K = len(S)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                refs_flat.append(S[i])
                hypos_flat.append(S[j])

    return corpus_bleu_sacre(hypos_flat, [refs_flat], tokenizer=tokenizer)

def leave_one_out_multi_ref_bleu(
    references: List[List[str]],
    all_predictions: Dict[int, List[str]],
    tokenizer: str = "none",
) -> float:
    """
    leave-one-out multi-reference corpus BLEU, as in Shen et al. (2019) and fairseq/examples/translation_moe/score.py.
    Steps:
      1) Transpose refs to shape [M refs][N examples]
      2) Transpose hypos to shape [K hyps][N examples]
      3) Flatten hypos across K to one long list over all examples (concatenate K streams).
      4) Duplicate each reference stream M times per hypothesis (so each reference list aligns with the flattened hypos).
      5) For each held-out reference m, remove it from the reference set and compute corpus BLEU; average over m.
    Returns averaged LOO BLEU (float).
    """
    # references: List over N examples, each is a List[str] of M references
    # all_predictions: {hyp_idx: List[str]} with length N per List
    N = len(references)
    Size_set = {len(r) for r in references}
    if len(Size_set) == 1 and Size_set.pop() == 1:
        return np.nan
    K = len(all_predictions)
    assert N > 0 and K > 0, "Empty inputs to LOO BLEU"

    # Transpose refs -> list(zip(*references)) gives M lists (each list is length N)
    refs_by_index = list(zip(*references))            # List[M][N]
    M = len(refs_by_index)
    # Transpose hypos -> K lists (each list is length N)
    hypos_by_h = [all_predictions[k] for k in range(K)]  # List[K][N]

    # Flatten hypos interleaving by hypothesis (to match Fairseq flattening)
    # flat_hypos[i_ex*K + k] = hypos_by_h[k][i_ex]
    flat_hypos: List[str] = [hypos_by_h[k][i] for i in range(N) for k in range(K)]

    # For each reference stream (length N), duplicate it K times to align with flat_hypos
    duplicated_refs: List[List[str]] = []
    for m_idx in range(M):
        refs_stream = refs_by_index[m_idx]            # length N
        # Repeat each ref exactly K times in the same per-example order
        duplicated = []
        for ref_sentence in refs_stream:
            duplicated.extend([ref_sentence] * K)
        duplicated_refs.append(duplicated)           # List[M][N*K]

    loo_scores = []
    for held_out in range(M):
        remaining = duplicated_refs[:held_out] + duplicated_refs[held_out + 1 :]
        # remaining is List[M-1][N*K]
        score = corpus_bleu_sacre(flat_hypos, remaining, tokenizer=tokenizer)
        loo_scores.append(score)

    return float(np.mean(loo_scores))

def refs_covered_metric(
    references: List[List[str]],
    all_predictions: Dict[int, List[str]],
    tokenizer: str = "13a",
) -> float:
    """
    Approximates how many distinct references get 'matched' per example by picking,
    for each hypothesis, the reference with highest sentence BLEU and counting unique winners.
    Mirrors the idea in fairseq's multi_ref() (but using SacreBLEU's sentence_bleu).
    Returns average count of distinct refs selected per example.
    """
    M = len(references[0])
    N = len(references)
    K = len(all_predictions)
    covered_total = 0.0
    for i in range(N):
        # collect the K hypos for this example
        hyps_i = [all_predictions[k][i] for k in range(K)]
        refs_i = references[i]
        chosen = set()
        for h in hyps_i:
            s = [sentence_bleu(h, [r], tokenize=tokenizer).score for r in refs_i]
            best_val = max(s)
            best_idxs = [j for j, v in enumerate(s) if v == best_val]
            # choose one of ties (deterministically pick first)
            chosen.add(best_idxs[0])
        covered_total += len(chosen)
    return covered_total / N

#####################

def compute_sentence_level_bleu_scores(
    predictions: List[str], 
    references: List[List[str]], 
    tokenizer: str = "13a"
) -> List[float]:
    """
    Compute sentence-level BLEU scores using official SacreBLEU library.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences (each item is a list of references for that prediction)
        tokenizer: Tokenizer to use ("13a" for German MT evaluation)
    
    Returns:
        List of sentence-level BLEU scores
    """
    bleu_scores = []
    
    for pred, refs in zip(predictions, references):
        # Use SacreBLEU's sentence_bleu function
        score = sentence_bleu(pred, refs, tokenize=tokenizer)
        bleu_scores.append(score.score)
    
    return bleu_scores

def compute_pairwise_bleu_fairseq_style(
    all_predictions: Dict[int, List[str]], 
    tokenizer: str = "13a"
) -> Dict[str, Any]:
    """
    Compute Pairwise-BLEU exactly following Fairseq MoE score.py implementation.
    
    For each hypothesis i, compute BLEU(hyp_i, all_other_hypotheses) where:
    - hyp_i is the candidate
    - all other hypotheses serve as multiple references
    - No symmetrization (as per actual Fairseq implementation)
    - Computes both sentence-level and corpus-level versions
    
    Based on: https://github.com/facebookresearch/fairseq/blob/main/examples/translation_moe/score.py
    
    Args:
        all_predictions: Dict mapping hypothesis_idx -> list of predictions
        tokenizer: Tokenizer to use
    
    Returns:
        Dictionary with pairwise BLEU scores and statistics
    """
    num_hypotheses = len(all_predictions)
    num_examples = len(list(all_predictions.values())[0])
    
    # Initialize BLEU metric
    bleu_metric = BLEU(tokenize=tokenizer)
    
    # Store pairwise BLEU scores for each hypothesis as candidate
    sentence_level_scores = {}
    corpus_level_scores = {}
    
    # For each hypothesis as candidate
    for cand_idx in range(num_hypotheses):
        candidate_predictions = all_predictions[cand_idx]
        
        # Get all other hypotheses as references
        reference_hypotheses = [all_predictions[ref_idx] for ref_idx in range(num_hypotheses) if ref_idx != cand_idx]
        
        if not reference_hypotheses:
            continue
        
        # === SENTENCE-LEVEL COMPUTATION ===
        sentence_scores = []
        for example_idx in range(num_examples):
            candidate_sent = candidate_predictions[example_idx]
            reference_sents = [ref_hyp[example_idx] for ref_hyp in reference_hypotheses]
            
            # Compute BLEU with multiple references (other hypotheses)
            score = sentence_bleu(candidate_sent, reference_sents, tokenize=tokenizer)
            sentence_scores.append(score.score)
        
        sentence_level_scores[cand_idx] = {
            'scores': sentence_scores,
            'mean': np.mean(sentence_scores),
            'std': np.std(sentence_scores),
            'num_references': len(reference_hypotheses)
        }
        
        # === CORPUS-LEVEL COMPUTATION ===
        # Prepare references in the format expected by corpus_score
        # corpus_score expects: candidates (list of str), list_of_references (list of list of str)
        references_per_example = []
        for example_idx in range(num_examples):
            refs_for_example = [ref_hyp[example_idx] for ref_hyp in reference_hypotheses]
            references_per_example.append(refs_for_example)
        
        # Compute corpus-level BLEU: one candidate list vs multiple reference lists
        corpus_score = bleu_metric.corpus_score(candidate_predictions, list(zip(*reference_hypotheses)))
        
        corpus_level_scores[cand_idx] = {
            'bleu_score': corpus_score.score,
            'bleu_1': corpus_score.precisions[0],
            'bleu_2': corpus_score.precisions[1], 
            'bleu_3': corpus_score.precisions[2],
            'bleu_4': corpus_score.precisions[3],
            'bp': corpus_score.bp,
            'ratio': corpus_score.ratio,
            'hyp_len': corpus_score.sys_len,
            'ref_len': corpus_score.ref_len,
            'num_references': len(reference_hypotheses)
        }
    
    # Compute overall statistics
    sentence_means = [scores['mean'] for scores in sentence_level_scores.values()]
    corpus_scores = [scores['bleu_score'] for scores in corpus_level_scores.values()]
    corpus_bleu4_scores = [scores['bleu_4'] for scores in corpus_level_scores.values()]
    
    results = {
        # Sentence-level results (averaged across sentences first, then across hypotheses)
        'sentence_level': {
            'mean_pairwise_bleu': float(np.mean(sentence_means)),
            # 'std_pairwise_bleu': float(np.std(sentence_means)),
            # 'per_hypothesis_scores': sentence_level_scores
        },
        
        # Corpus-level results (computed at corpus level for each hypothesis, then averaged)
        'corpus_level': {
            'mean_pairwise_bleu': float(np.mean(corpus_scores)),
            'mean_pairwise_bleu4': float(np.mean(corpus_bleu4_scores)),  # mBLEU-4 equivalent
            # 'std_pairwise_bleu': float(np.std(corpus_scores)),
            # 'per_hypothesis_scores': corpus_level_scores
        },
        
        # For backward compatibility
        # 'mean_pairwise_bleu': float(np.mean(sentence_means)),  # Default to sentence-level
        
        # Metadata
        'num_hypotheses': num_hypotheses,
        'num_examples': num_examples
    }
    
    return results

def compute_pairwise_diversity(
    all_predictions: Dict[int, List[str]], 
    tokenizer: str = "13a"
) -> Dict[str, float]:
    """
    Compute diversity metrics between hypothesis pairs.
    
    Args:
        all_predictions: Dict mapping hypothesis_idx -> list of predictions
        tokenizer: Tokenizer to use
    
    Returns:
        Dictionary with diversity metrics
    """
    num_hypotheses = len(all_predictions)
    num_examples = len(list(all_predictions.values())[0])
    
    # Compute pairwise BLEU between all hypothesis pairs
    pairwise_bleu_matrix = np.zeros((num_hypotheses, num_hypotheses))
    
    for i, j in combinations(range(num_hypotheses), 2):
        predictions_i = all_predictions[i]
        predictions_j = all_predictions[j]
        
        # Compute BLEU between hypothesis i and j
        bleu_i_j = []
        bleu_j_i = []
        
        for example_idx in range(num_examples):
            # i as candidate, j as reference
            score_i_j = sentence_bleu(predictions_i[example_idx], [predictions_j[example_idx]], tokenize=tokenizer)
            bleu_i_j.append(score_i_j.score)
            
            # j as candidate, i as reference
            score_j_i = sentence_bleu(predictions_j[example_idx], [predictions_i[example_idx]], tokenize=tokenizer)
            bleu_j_i.append(score_j_i.score)
        
        # Store symmetric average
        avg_bleu = (np.mean(bleu_i_j) + np.mean(bleu_j_i)) / 2
        pairwise_bleu_matrix[i, j] = avg_bleu
        pairwise_bleu_matrix[j, i] = avg_bleu
    
    # Set diagonal to 100 (self-BLEU)
    np.fill_diagonal(pairwise_bleu_matrix, 100.0)
    
    # Compute diversity metrics
    # Lower BLEU between hypotheses indicates higher diversity
    off_diagonal_values = pairwise_bleu_matrix[np.triu_indices(num_hypotheses, k=1)]
    
    diversity_metrics = {
        'mean_pairwise_bleu': np.mean(off_diagonal_values),
        "num_hypotheses": num_hypotheses,
        "num_examples": num_examples
    }
    
    return diversity_metrics

def compute_oracle_sentence_bleu(
    all_predictions: Dict[int, List[str]], 
    references: List[List[str]], 
    tokenizer: str = "13a"
) -> Dict[str, Any]:
    """
    Compute oracle sentence-level BLEU scores across multiple hypotheses.

        Args:
        all_predictions: Dict mapping hypothesis_idx -> list of predictions
        references: List of reference sentences for each example
        tokenizer: Tokenizer to use

        Returns:
        Dictionary with oracle scores and per-hypothesis scores
    """
    num_examples = len(references)
    num_hypotheses = len(all_predictions)
    
    # Compute BLEU scores for each hypothesis
    hypothesis_scores = {}
    for hyp_idx, predictions in all_predictions.items():
        scores = compute_sentence_level_bleu_scores(predictions, references, tokenizer)
        hypothesis_scores[hyp_idx] = scores
    
    # Compute oracle scores (best score per sentence)
    oracle_scores = []
    oracle_hypothesis_indices = []
    
    for example_idx in range(num_examples):
        example_scores = [hypothesis_scores[hyp_idx][example_idx] for hyp_idx in range(num_hypotheses)]
        best_score = max(example_scores)
        best_hypothesis = example_scores.index(best_score)
        
        oracle_scores.append(best_score)
        oracle_hypothesis_indices.append(best_hypothesis)
    
    return {
        'oracle_mean': sum(oracle_scores) / len(oracle_scores),
    }

def load_and_process_data(path: str, num_hypotheses: int) -> Tuple[List[List[str]], Dict[int, List[str]]]:
    """
    Load and process the pickle file to extract references and predictions.
    
    Args:
        path: Path to pickle file
        num_hypotheses: Number of hypotheses
    
    Returns:
        Tuple of (references, all_predictions)
    """
    with open(path, "rb") as f:
        outputs = pickle.load(f)
    
    references = []
    all_predictions = {}
    sources = None
    
    # Extract predictions and references from your data structure
    for hypothesis_idx in range(num_hypotheses):
        hypothesis_key = f"hypothesis_{hypothesis_idx}"
        if hypothesis_key in outputs:
            # Extract predictions for this hypothesis
            predictions = outputs[hypothesis_key].get('cands', [])
            all_predictions[hypothesis_idx] = predictions
            
            # Extract references (only need to do this once)
            if hypothesis_idx == 0:
                raw_refs = outputs[hypothesis_key].get('mrefs', [])
                # Convert to the format expected by sentence_bleu (list of lists)
                references = [[ref] if isinstance(ref, str) else ref for ref in raw_refs]
                maybe_src = outputs[hypothesis_key].get("src")
                if isinstance(maybe_src, list) and all(isinstance(s, str) for s in maybe_src):
                    sources = maybe_src


    return references, all_predictions, sources

def main(path: str, num_hypotheses: int):
    """
    Main function implementing sentence-level BLEU, Oracle BLEU, and Pairwise-BLEU evaluation.
    """    
    
    # Initialize BLEU metric with 13a tokenizer
    bleu_metric = BLEU(tokenize="13a")
    
    # Load and process data
    references, all_predictions, sources = load_and_process_data(path, num_hypotheses)

    # Detokenize references and all predictions
    for h in list(all_predictions.keys()):
        all_predictions[h] = detok_lines(all_predictions[h], scheme='moses', lang='de')
    references = detok_nested_refs(references, scheme='moses', lang='de')

    if len(all_predictions[0]) < len(references):
        references = references[:len(all_predictions[0])]
    
    print(f"Loaded {len(references)} examples with {num_hypotheses} hypotheses each")
    
    # 1. Compute oracle sentence-level BLEU scores
    print("Computing Oracle BLEU scores...")
    try:
        oracle_results = compute_oracle_sentence_bleu(
        all_predictions, 
        references, 
        tokenizer="13a"
    )
    except Exception as e:
        print(f"[WARN] Oracle BLEU computation skipped due to error: {e}")
        traceback.print_exc()
        oracle_results = None
    
    # 2. Compute pairwise BLEU scores (Fairseq MoE style)
    print("Computing Pairwise-BLEU scores")
    try:
        pairwise_results = compute_pairwise_bleu_fairseq_style(
        all_predictions, 
        tokenizer="13a"
    )
    except Exception as e:
        print(f"[WARN] Pairwise BLEU computation skipped due to error: {e}")
        traceback.print_exc()
        pairwise_results = None
    
    # 3. Compute corpus-level BLEU for comparison
    print("Computing corpus-level BLEU scores...")
    try:
        corpus_bleu_scores = {}
        for hyp_idx, predictions in all_predictions.items():
            # Flatten references for corpus-level computation
            flat_refs = [ref[0] if isinstance(ref, list) and len(ref) > 0 else ref for ref in references]
            
            # Use SacreBLEU's corpus_bleu
            corpus_score = bleu_metric.corpus_score(predictions, [flat_refs])
            corpus_bleu_scores[f'hypothesis_{hyp_idx}'] = corpus_score.score
    except Exception as e:
        print(f"[WARN] Corpus BLEU computation skipped due to error: {e}")
        traceback.print_exc()
        corpus_bleu_scores = None
        
    
    # 4. Compute Fairseq-style Pairwise-BLEU scores
    print("Computing Fairseq-style Pairwise-BLEU scores")
    try:
        fairseq_pairwise_bleu = pairwise_bleu_fairseq_exact(all_predictions, tokenizer="none")  # use "13a" if NOT pre-tokenized
    except Exception as e:
        print(f"[WARN] Fairseq-style Pairwise BLEU computation skipped due to error: {e}")
        traceback.print_exc()
        fairseq_pairwise_bleu = None

    # Leave-one-out multi-reference BLEU (averaged over M held-outs)
    print("Computing Fairseq-style Leave-one-out multi-reference BLEU scores")
    try:
        loo_bleu = leave_one_out_multi_ref_bleu(references, all_predictions, tokenizer="none")
    except Exception as e:
        print(f"[WARN] Leave-one-out multi-reference BLEU computation skipped due to error: {e}")
        traceback.print_exc()
        loo_bleu = None

    print("Doing the same but with 13a tokenizer")
    try:
        fairseq_pairwise_bleu_13a = pairwise_bleu_fairseq_exact(all_predictions, tokenizer="13a")  # use "13a" if NOT pre-tokenized
    except Exception as e:
        print(f"[WARN] Fairseq-style Pairwise BLEU computation skipped due to error: {e}")
        traceback.print_exc()
        fairseq_pairwise_bleu_13a = None

    print("Computing Fairseq-style Pairwise-BLEU scores")
    try:
        loo_bleu_13a = leave_one_out_multi_ref_bleu(references, all_predictions, tokenizer="13a")
    except Exception as e:
        print(f"[WARN] Leave-one-out multi-reference BLEU computation skipped due to error: {e}")
        traceback.print_exc()
        loo_bleu_13a = None

    # refs covered
    try:
        refs_cov = refs_covered_metric(references, all_predictions, tokenizer="13a")
    except Exception as e:
        print(f"[WARN] Refs covered metric computation skipped due to error: {e}")
        traceback.print_exc()
        refs_cov = None

    # Prepare comprehensive results
    results = {
        # Oracle evaluation
        'oracle_sentence_bleu_mean': oracle_results['oracle_mean'] if oracle_results is not None else None,
        
        # Pairwise BLEU evaluation (Fairseq style - no symmetrization)
        'pairwise_bleu_sentence_level': pairwise_results['sentence_level'] if pairwise_results is not None else None,
        'pairwise_bleu_corpus_level': pairwise_results['corpus_level'] if pairwise_results is not None else None,
        # Corpus-level BLEU
        'corpus_bleu_scores': corpus_bleu_scores,
        
        # Metadata
        'num_hypotheses': num_hypotheses,
        'num_examples': len(references),

        'fairseq_pairwise_bleu': fairseq_pairwise_bleu,
        'fairseq_loo_bleu': loo_bleu,
        'fairseq_refs_covered': refs_cov,
        'fairseq_pairwise_bleu_13a': fairseq_pairwise_bleu_13a,
        'fairseq_loo_bleu_13a': loo_bleu_13a,
    }

    ### Comet options ###
    # --- COMET (optional) ---
    # read CLI from global namespace (quick way without refactoring),
    # or pass as parameters if you prefer cleaner structure.
    try:
        # Access sys.argv parsed once at module load; if you prefer, pass args explicitly
        import argparse
        # Reparse minimal flags safely (or restructure to pass args into main)
        # Here, we detect presence conservatively:
        # do_comet = any(a == "--comet" for a in sys.argv)
        do_comet = False
        if do_comet:
            model_name = "Unbabel/wmt22-comet-da"
            bs = 32
            force_cpu = False
            # light parsing for overrides
            for i, a in enumerate(sys.argv):
                if a == "--comet_model" and i + 1 < len(sys.argv):
                    model_name = sys.argv[i + 1]
                if a == "--comet_batch_size" and i + 1 < len(sys.argv):
                    bs = int(sys.argv[i + 1])
                if a == "--comet_cpu":
                    force_cpu = True

            comet_block = compute_comet_for_all_hypotheses(
                references=references,
                all_predictions=all_predictions,
                sources=sources,
                model_name=model_name,
                batch_size=bs,
                use_gpu=not force_cpu
            )
            results["comet"] = comet_block
            print("Computed COMET and oracle-COMET.")
    except Exception as e:
        print(f"[WARN] COMET computation skipped due to error: {e}")
        traceback.print_exc()  # <-- full stack trace
    
    # Save results
    save_folder = os.path.dirname(path)
    os.makedirs(save_folder, exist_ok=True)
    
    output_file = os.path.join(save_folder, os.path.basename(path).replace(".pkl", "_FIXED_metrics_DETOK.yaml"))
    print(f"Saving comprehensive BLEU metrics to {output_file}")
    
    with open(output_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    ### Save predictions in a yaml file
    output_file_predictions = os.path.join(save_folder, os.path.basename(path).replace(".pkl", "_FIXED_metrics_DETOK_predictions.yaml"))
    print(f"Saving predictions to {output_file_predictions}")
    with open(output_file_predictions, "w") as f:
        yaml.dump(all_predictions, f, default_flow_style=False)

    ### Save references in a yaml file
    output_file_references = os.path.join(save_folder, os.path.basename(path).replace(".pkl", "_FIXED_metrics_DETOK_references.yaml"))
    print(f"Saving references to {output_file_references}")
    with open(output_file_references, "w") as f:
        yaml.dump(references, f, default_flow_style=False)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive BLEU evaluation with Oracle, Pairwise, and standard metrics")
    parser.add_argument("--path", type=str, required=True, help="Path to predictions pickle file")
    parser.add_argument("--num_hypotheses", type=int, required=True, help="Number of hypotheses per example")
   
    args = parser.parse_args()

    main(args.path, args.num_hypotheses)
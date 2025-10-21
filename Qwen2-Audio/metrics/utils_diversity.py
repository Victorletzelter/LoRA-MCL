import sys
import os
import numpy as np
from functools import partial
from multiprocessing import Pool
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os, sys
sys.path.append(os.path.join(os.environ["COCO_CAPTION_PATH"],'..'))
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

#%%

def eval_div_stats(preds_n, compute_mbleu=True):
    capsById = preds_n
    tokenizer = PTBTokenizer()
    n_caps_perimg = len(capsById[list(capsById.keys())[0]])
    # n_caps_perimg = len(capsById[0])
    _capsById = capsById # save the untokenized version
    capsById = tokenizer.tokenize(capsById)
    div_1, adiv_1 = compute_div_n(capsById,1)
    div_2, adiv_2 = compute_div_n(capsById,2)

    globdiv_1, _ = compute_global_div_n(capsById,1)

    if compute_mbleu is True: 
        # compute mbleu
        scorer = Bleu(4)
        all_scrs = []
        scrperimg = np.zeros((n_caps_perimg, len(capsById), 4))

        for i in range(n_caps_perimg):
            tempRefsById = {}
            candsById = {}
            for k in capsById:
                tempRefsById[k] = capsById[k][:i] + capsById[k][i+1:]
                candsById[k] = [capsById[k][i]]

            score, scores = scorer.compute_score(tempRefsById, candsById)
            all_scrs.append(score)
            for p in range(4):
                scrperimg[i,:,p] = scores[p]

        all_scrs = np.array(all_scrs)
        
        out = {}
        out['overall'] = {'Div1': div_1, 'Div2': div_2, 'gDiv1': globdiv_1}
        for k, score in zip(range(4), all_scrs.mean(axis=0).tolist()):
            out['overall'].update({'mBLeu_%d'%(k+1): score})

        return globdiv_1,div_1,div_2,out['overall']['mBLeu_4'], scrperimg, adiv_1, adiv_2
    else:
        return globdiv_1,div_1,div_2,None,None,adiv_1, adiv_2 

def calc_ngram(words, n=2):
    return zip(*[words[i:] for i in range(n)])

def calc_self_bleu(sentences, num_workers):

    pool = Pool(num_workers)
    result = []
    for idx in range(len(sentences)):
        hypothesis = sentences[idx]
        references = [sentences[_] for _ in range(len(sentences)) if _ != idx]
        result.append(pool.apply_async(
            partial(sentence_bleu, smoothing_function=SmoothingFunction().method1),
            args=(references, hypothesis))
        )
    score = 0.0
    cnt = 0
    for i in result:
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def compute_div_n(caps,n=1):
    aggr_div = []
    for k in caps:
        all_ngrams = set()
        lenT = 0.
        for c in caps[k]:
            tkns = c.split()
            # print(tkns)
            lenT += len(tkns)
            # print(lenT)
            ng = find_ngrams(tkns, n)
            all_ngrams.update(ng)
            # break
        # print(len(all_ngrams))
        
        aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT)))
        # print(aggr_div)
        # break
    return np.array(aggr_div).mean(), np.array(aggr_div)

def compute_global_div_n(caps,n=1):
  aggr_div = []
  all_ngrams = set()
  lenT = 0.
  for k in caps:
      for c in caps[k]:
         tkns = c.split()
         lenT += len(tkns)
         ng = find_ngrams(tkns, n)
         all_ngrams.update(ng)
  if n == 1:
    aggr_div.append(float(len(all_ngrams)))
  else:
    aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT)))
  return aggr_div[0], np.repeat(np.array(aggr_div),len(caps))
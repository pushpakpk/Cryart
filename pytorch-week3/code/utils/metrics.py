from nltk.translate.bleu_score import corpus_bleu

def compute_bleu(references, hypotheses):
    return corpus_bleu(references, hypotheses)*100

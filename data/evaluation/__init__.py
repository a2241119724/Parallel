from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .spice import Spice

def compute_scores(gts, gen, isSpice=False):
    if isSpice:
        spice = Spice()
        score, scores = spice.compute_score(gts, gen)
        all_score[str(spice)] = score
        all_scores[str(spice)] = scores
        return all_score, all_scores
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    # metrics = [Cider()]
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    return all_score, all_scores

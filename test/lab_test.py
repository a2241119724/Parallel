import os
import sys
sys.path.append(os.path.abspath("./"))

# from pycocoevalcap.cider.cider import Cider
from data.evaluation.cider import Cider
from collections import defaultdict

# 假设这是你的预测和参考
preds = {'391895': ['a man with a red helmet on a small moped on a dirt road']}
refs = {'391895': ['a man in a kitchen holding a teddy bear']}

# 初始化CIDEr评估器
cider_scorer = Cider()

# 计算CIDEr分数
score, scores = cider_scorer.compute_score(refs, preds)

# 将分数转换为字典，以便于查看每个预测的CIDEr分数
scores_dict = defaultdict(list)
for pred_id, score in zip(preds.keys(), scores):
    scores_dict[pred_id].append(score)

print(scores_dict)
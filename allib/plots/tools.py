from operator import itemgetter

from ..metrics import get_metrics
from ..models.al import ActiveLearningMetric, ActiveLearningPipeline
from ..datasets import Dataset, load_uci


def make_recipt(dss: list, alms: list, ms: list, conf: dict):
    ppls = {}
    (ppl_makers,
     ppl_confs,
     ds_confs,
     alm_confs,
     model_confs) = itemgetter(
        "ppl_makers",
        "ppl_confs",
        "ds_confs",
        "alm_conf",
        "model_confs")(conf)
    ds_l = []
    alm_ctors = get_metrics(alms)
    for ds_idx, ds in enumerate(dss):
        (data, label), (tx, ty), cat_idx = load_uci(ds)
        for al_idx, alm_name in enumerate(alms):
            alc = alm_ctors[al_idx]
            alm = alc(**alm_confs[al_idx], **ds_confs[ds_idx])
            dataset = Dataset(data, label, al_metric=alm, shuffle=False, **ds_confs[ds_idx])
            for m_idx, m in enumerate(ms):
                ppls[(ds, alm_name, m)] = ppl_maker(**ppl_conf, **model_confs[m_idx])


def draw_boundary():
    pass
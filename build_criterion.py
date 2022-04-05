import criterions


def get_criterion(cfg=None):

    if cfg.criterion == "rouge":
        rouge = criterions.get_rouge_metric()
        return rouge
    else:
        print(f" No loss criterion specified")
        return None

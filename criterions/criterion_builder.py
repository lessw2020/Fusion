from nlp import load_metric


def get_rouge_metric():
    rouge_metric = None
    rouge_metric = load_metric("rouge")
    if rouge_metric is None:
        print(f"Failed to load rouge metric!")
        RaiseValueError("unable to load metric")
    else:
        print(f"--> Criterion Rouge loaded")

    return rouge_metric

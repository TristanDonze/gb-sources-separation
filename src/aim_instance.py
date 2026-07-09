from aim import Run

aim_run = Run(experiment="gb-sources-separation")

def track_metric(
    name,
    value,
    *,
    step=None,
    epoch=None,
    split=None,
    granularity=None,
    predictor=None,
):
    context = {}
    if split is not None:
        context["split"] = split
    if granularity is not None:
        context["granularity"] = granularity
    if predictor is not None:
        context["predictor"] = predictor

    aim_run.track(
        float(value),
        name=name,
        step=step,
        epoch=epoch,
        context=context,
    )

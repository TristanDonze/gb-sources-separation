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
    source_count=None,
):
    context = {}
    if split is not None:
        context["split"] = split
    if granularity is not None:
        context["granularity"] = granularity
    if source_count is not None:
        context["K"] = int(source_count)

    aim_run.track(
        float(value),
        name=name,
        step=step,
        epoch=epoch,
        context=context,
    )

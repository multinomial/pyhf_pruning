# +
from copy import deepcopy

def prune_model(workspace, eps=0.001):
    workspace = deepcopy(workspace)
    channels = workspace.get("channels")

    for channel_id, channel in enumerate(channels):
        samples = channel.get("samples")

        for sample_id, sample in enumerate(samples):
            modifiers = sample.get("modifiers")

            ids_to_remove = []

            for modifier_id, modifier in enumerate(modifiers):
                data = modifier.get("data")
                if (modifier.get("type") == "normsys") and (
                    max(abs(1 - data["hi"]), abs(1 - data["lo"])) <= eps
                ):
                    ids_to_remove.append(modifier_id)

            for modifier_id in ids_to_remove[::-1]:
                del workspace["channels"][channel_id]["samples"][sample_id][
                    "modifiers"
                ][modifier_id]

    return workspace

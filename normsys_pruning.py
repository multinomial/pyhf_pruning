import json
import numpy as np
import pyhf


def prune_model(workspace, eps=0.001):

    channels = workspace.get("channels")

    for channel_id, channel in enumerate(channels):
        samples = channel.get("samples")

        for sample_id, sample in enumerate(samples):
            modifiers = sample.get("modifiers")

            ids_to_remove = []

            for modifier_id, modifier in enumerate(modifiers):
                data = modifier.get("data")
                if (modifier.get("type") == "normsys") and (
                    np.max(np.abs(1 - data["hi"]), np.abs(1 - data["lo"])) <= eps
                ):
                    ids_to_remove.append(modifier_id)

            for modifier_id in ids_to_remove[::-1]:
                del workspace["channels"][channel_id]["samples"][sample_id][
                    "modifiers"
                ][modifier_id]

    return workspace


path = "bkg_only.json"
path2 = "example_workspace.json"


input_file = open(path, "r")
workspace = json.load(input_file)

new_workspace = prune_model(workspace)

output_file = open(path[:-5:] + "_pruned.json", "w")
json.dump(new_workspace, output_file)

input_file.close()
output_file.close()

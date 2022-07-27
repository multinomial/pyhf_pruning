import json
import numpy as np
import pyhf


def prune_histosys(workspace, eps=0.01):

    channels = workspace.get("channels")

    for channel_id, channel in enumerate(channels):
        samples = channel.get("samples")

        for sample_id, sample in enumerate(samples):
            modifiers = sample.get("modifiers")
            data = np.array(sample.get("data"))

            ids_to_remove = []

            for modifier_id, modifier in enumerate(modifiers):
                if modifier.get("type") == "histosys":
                    mod_data = modifier.get("data")

                    if (
                        np.mean(
                            np.max(
                            np.vstack(
                                (
                                np.abs(np.array(mod_data["hi_data"]) - data),
                                np.abs(np.array(mod_data["lo_data"]) - data)
                                )
                            ),
                             axis = 0
                            )
                            / data
                        )
                        <= eps
                       ):
                        ids_to_remove.append(modifier_id)

            for modifier_id in ids_to_remove[::-1]:
                print("pruned modifier {}, in channel {}, sample {}".format(modifier_id, channel_id, sample_id))
                del workspace["channels"][channel_id]["samples"][sample_id][
                    "modifiers"
                ][modifier_id]

    return workspace


path = "bkg_only.json"

# +
input_file = open(path, 'r')
workspace = json.load(input_file)

new_workspace = prune_histosys(workspace, 0.001)

output_file = open(path[:-5:] + "_pruned.json", "w")
json.dump(new_workspace, output_file)

input_file.close()
output_file.close()

import json
import numpy as np
import pyhf


def prune_histosys(workspace, eps=0.1):

    channels = workspace.get("channels")

    for channel_id, channel in enumerate(channels):
        samples = channel.get("samples")

        for sample_id, sample in enumerate(samples):
            modifiers = sample.get("modifiers")
            data = sample.get("data")

            ids_to_remove = []

            for modifier_id, modifier in enumerate(modifiers):
                if modifier.get("type") == "histosys":
                    mod_data = modifier.get("data")

                    if (
                        np.mean(
                            np.max(
                                np.abs(mod_data["hi_data"] - data),
                                np.abs(mod_data["lo_data"] - data)
                            )
                            / data
                        )
                        <= eps
                    ):
                        ids_to_remove.append(modifier_id)

            for modifier_id in ids_to_remove[::-1]:
                del workspace["channels"][channel_id]["samples"][sample_id][
                    "modifiers"
                ][modifier_id]

    return workspace

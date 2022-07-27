import json
import numpy as np
import pyhf


def prune_histosys(workspace, eps=0.1):
    return 0


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

                        workspace["channels"][channel_id]["samples"][sample_id]["modifiers"][modifier_id]["data"]["hi_data"] = np.where(
                            np.max(
                                np.abs(mod_data["hi_data"] - data),
                                np.abs(mod_data["lo_data"] - data)
                            )
                            / data 
                            <= eps,
                            data,
                            mod_data["hi_data"]
                        )
                        
                        workspace["channels"][channel_id]["samples"][sample_id]["modifiers"][modifier_id]["data"]["lo_data"] = np.where(
                            np.max(
                                np.abs(mod_data["hi_data"] - data),
                                np.abs(mod_data["lo_data"] - data)
                            )
                            / data 
                            <= eps,
                            data,
                            mod_data["lo_data"]
                        )


    return workspace

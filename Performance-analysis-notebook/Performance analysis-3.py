#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pyhf
import json
import cabinetry
from os import listdir, getcwd
import requests
from jsonschema import validate
import pruning.normsys_pruning as pruning
#from os.path import isfile, isdir, join


# In[ ]:


path = getcwd()
filename = "/workspace_Comb.json"


# In[ ]:


workspace_spec = json.load(open(path + filename, 'r'))

workspace = pyhf.Workspace(workspace_spec) 


# In[ ]:


original_model = workspace.model()


# In[ ]:


num_thresholds = 5
pruning_thresholds = [float(eps) for eps in np.linspace(0.01, 0.0, num_thresholds)] 
pruning_thresholds


# In[ ]:


pruned_workspaces_specs = [ pruning.prune_model(workspace_spec, eps) for eps in pruning_thresholds ]


# In[ ]:


for l in range(num_thresholds):
    output_file = open(path + "/spec_{}.json".format(pruning_thresholds[l]), "w")
    json.dump(pruned_workspaces_specs[l], output_file, indent=4)
    output_file.close()


# In[ ]:


pruned_workspaces = [ pyhf.Workspace(workspace_spec) for workspace_spec in pruned_workspaces_specs ]


# In[ ]:


pruned_models = [ pruned_workspace.model() for pruned_workspace in pruned_workspaces ]


# In[ ]:


import time

num_executions = 1


# In[ ]:


pyhf.set_backend("numpy", "minuit")


# In[ ]:


get_ipython().run_line_magic('time', '')
average_exec_times = []
exec_times = []
output_params_all_specs = []

for l in range(num_thresholds):
    
    pruned_data = pruned_workspaces[l].data(pruned_models[l]) #, include_auxdata=False)

    exec_times_pruned = []
    output_params_sigle_spec = []

    for k in range(num_executions):
        t0 = time.time()
        output_params = pyhf.infer.mle.fit(data=pruned_data, pdf=pruned_models[l])
        t1 = time.time()
        exec_times_pruned.append(t1-t0)
        
        output_params_sigle_spec.append(dict(zip(pruned_models[l].config.par_names(), output_params)))
        print(k+1)
        print(output_params)
        
    print("----------------------------")
    exec_times_pruned = np.array(exec_times_pruned)
    
    exec_times.append(exec_times_pruned)
    average_exec_times.append(np.mean(exec_times_pruned))
    
    output_params_all_specs.append(output_params_sigle_spec)
    


# In[ ]:


# for k in range(num_executions):
    for l in range(num_thresholds-1):
        pruned_output_params_dict = output_params_all_specs[l][k]
        parameter_names = pruned_output_params_dict.keys()
        
        pruned_output_params = np.fromiter(pruned_output_params_dict.values(), float)
        original_output_params = np.array([output_params_all_specs[num_thresholds-1][k][name] for name in parameter_names])
        print(np.abs(pruned_output_params - original_output_params))
        #print(pruned_output_params)
        #print(original_output_params)
    print("-------------------------------")


# In[ ]:


results = {"num_executions" : num_executions, "num_thresholds" : num_thresholds, "pruning_thresholds" : pruning_thresholds, "average_exec_times" : average_exec_times, "exec_times" : exec_times, "output_params_all_specs" : output_params_all_specs }


# In[ ]:


output_file = open(path + filename[:-5:] + "_PA_results.json", "w")
json.dump(results, output_file, indent=4)
output_file.close()


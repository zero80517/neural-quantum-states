import numpy as np
import json


def encode(z):
    if isinstance(z, complex):
        return [z.real, z.imag]
    if isinstance(z, np.int32):
        return int(z)
    if isinstance(z, np.int64):
        return int(z)
    type_name = z.__class__.__name__
    raise TypeError(f"Object of type '{type_name}' is not JSON serializable")
    

def get_data(vmc):
    log = dict()
    log.update(vmc.sampler.get_params() )
    log.update(vmc.optimizer.get_params() )
    log['training'] = list()

    for i in range(len(vmc.sampler_results) ):
        element = dict()
        element.update(vmc.sampler_results[i])
        element.update(vmc.optimizer_results[i])
        element.update({'iteration': i+1})
        log['training'].append(element)

    nqs = dict()
    nqs.update(vmc.operator.get_params())
    nqs.update(vmc.nqs.get_params())
    nqs.update({'samples': vmc.sampler.get_samples()})

    data = {'nqs': nqs, 'log': log}

    return data
    
    
def save_data(vmc):
        data = get_data(vmc)
        with open(vmc.filename + '.nqs', 'w') as write_file:
            json.dump(data['nqs'], write_file, default=encode)
        with open(vmc.filename + '.log', 'w') as write_file:
            json.dump(data['log'], write_file, default=encode)

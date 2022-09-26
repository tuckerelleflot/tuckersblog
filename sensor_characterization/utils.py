import pickle as pkl

def load_pkl(fname):

    try:
        with open(fname) as f:
            data = pkl.load(f)
        return data
    except: pass

    try:
        with open(fname, 'rb') as f:
            bdata = pkl.load(f, encoding='bytes')
        data = convert(bdata)
        return data
    except: pass

    try:
        with open(fname, 'rb') as f:
            bdata = pkl.load(f, encoding='latin1')
        data = convert(bdata)
        return data
    except: pass
    
    print('failed to load pkl')
    
def convert(data):
    # converts dictionary keys from bytes to strings
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(convert, data.items()))
    if isinstance(data, tuple):  return map(convert, data)
    return data 
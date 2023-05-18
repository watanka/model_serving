import yaml

def read_yaml(cfgfile) :
    with open(cfgfile, 'r') as f :
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg
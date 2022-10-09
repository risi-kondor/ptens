def linmaps0(x):
    return x.linmaps0()
    
def linmaps1(x):
    return x.linmaps1()
    
def linmaps2(x):
    return x.linmaps2()


def transfer0(x,_atoms,G):
    return x.transfer0(_atoms,G)
    
def transfer1(x,_atoms,G):
    return x.transfer1(_atoms,G)
    
def transfer2(x,_atoms,G):
    return x.transfer2(_atoms,G)


def cat(x,y):
    return x.concat(y)


def relu(x,alpha=0.5):
    return x.relu(alpha)


def unite1(x,G):
    return x.unite1(G)

def unite2(x,G):
    return x.unite2(G)

def gather(x,G):
    return x.gather(G)


def device_id(device):
    if device==0:
        return 0
    if device==1:
        return 1
    if device=='cpu':
        return 0
    if device=='cuda':
        return 1
    if device=='cuda:0':
        return 1
    return 0

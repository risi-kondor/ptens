limport torch

from ptens_base import ptensors1 as _ptensors1


class ptensors1(torch.Tensor):

    @classmethod
    def raw(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.raw(_atoms,_nc,_dev)

    @classmethod
    def zero(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.zero(_atoms,_nc,_dev)

    @classmethod
    def sequential(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.sequential(_atoms,_nc,_dev)


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=rtensorvar(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=rtensorvar(1)
        R.obj=self.obj.view_of_grad()
        return R


    def get_nc(self):
        return self.obj.get_nc()

    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def push_back(self, x):
        return self.obj.push_back(x)



    
    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()




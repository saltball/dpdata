from io import FileIO
from re import T
import numpy as np

try:
    from ase.units import Hartree, Bohr
    AU_TO_EV = Hartree
    Bh_TO_ANG = Bohr
except ImportError:
    AU_TO_EV = 27.211386024367243  # data from ase
    Bh_TO_ANG = 0.5291772105638411  # data from ase


class XTBXyzSystems:
    """
    deal with xTB output stream
    """

    def __init__(self, stdout_file_name=None, opt_file_name=None, opt_traj_file_name=None, grad_file_name=None, hessian_file_name=None):
        """
        Practically,
        `stdout_file_name` should be a file storing stdout stream with `>` when calling `xTB`,
        `opt_file_name` should be `xtbopt.xyz`, (--opt)
        # `opt_traj_name` should be `xtbopt.log`, (--opt)
        `grad_file_name` should be `gradient`, (--grad)
        `hessian_file_name` should be `hessian`, (--hess or --ohess)
        For now, other files like `g98.out`, `<input_name>.engrad`, `xtbout.json`,
                                  `wbo`, `charges` will not be transfer to inherent data.
        # TODO: decouple `opttraj` from this.
        """
        self.xtb_task_type = list()
        if stdout_file_name:
            self.xyz_file_object = open(stdout_file_name, 'r')
            self.xtb_task_type.append("output")
        if opt_file_name:
            self.opt_file_object = open(opt_file_name, "r")
            self.xtb_task_type.append("opt")
        if opt_traj_file_name:
            raise RuntimeError(f"{opt_traj_file_name} will not be proceed.")
        #     self.opt_traj_file_object = open(opt_traj_file_name, "r")
        #     self.xtb_task_type.append("opttraj")
        if grad_file_name:
            self.grad_file_object = open(grad_file_name, "r")
            self.xtb_task_type.append("grad")
        if hessian_file_name:
            self.hessian_file_object = open(hessian_file_name, "r")
            self.xtb_task_type.append("hess")
        if len(self.xtb_task_type):
            raise RuntimeWarning(
                f"No data in {self.__class__.__name__}, this will not stop.")

    # def __iter__(self):
    #     if "opttraj" in self.xtb_task_type:
    #         return self
    #     else:
    #         raise TypeError(
    #             f"{self.__class__.__name__} instance without `opttraj` data is not Iterable.")

    # def __next__(self):
    #     if "opttraj" in self.xtb_task_type:
    #         return self.handle_traj_frame(next(self.block_generator))
    #     else:
    #         raise TypeError(
    #             f"{self.__class__.__name__} instance without `opttraj` data is not Iterable.")

    def __del__(self):
        if getattr(self, "xyz_file_object"):
            self.xyz_file_object.close()
        if getattr(self, "opt_file_object"):
            self.opt_file_object.close()
        # if getattr(self, "opt_traj_file_object"):
        #     self.opt_traj_file_object.close()
        if getattr(self, "grad_file_object"):
            self.grad_file_object.close()
        if getattr(self, "hessian_file_object"):
            self.hessian_file_object.close()


def get_grad_frame(
        grad_file_obj):

    # refer to ase.io.turbomole, change some logic
    if isinstance(FileIO, grad_file_obj):
        grad_file_obj = open(grad_file_obj, "r")
    lines = [x.strip() for x in grad_file_obj.readlines()]
    start = end = -1
    for i, line in enumerate(lines):
        if not line.startswith('$'):
            continue
        if line.split()[0] == '$grad':
            start = i
        elif start >= 0:
            end = i
            break
    if end <= start:
        raise RuntimeError('File does not contain a valid \'$grad\' section')

    del lines[:start + 1]
    del lines[end - 1 - start:]
    while lines:  # loop over optimization cycles
        # header line
        # cycle =      {N}    SCF energy =     -{FLOAT}   |dE/dxyz| =  {FLOAT}
        fields = lines[0].split('=')
        try:
            # cycle = int(fields[1].split()[0])
            energy = float(fields[2].split()[0]) * AU_TO_EV
            # gradient = float(fields[3].split()[0])
        except (IndexError, ValueError) as e:
            raise e
        for line in lines[1:]:
            fields = line.split()
            if len(fields) == 4:  # coordinates
                # 0.00000000000000      0.00000000000000      0.00000000000000      X
                try:
                    symbol = fields[3].lower().capitalize()
                    # if dummy atom specified, substitute 'Q' with 'X'
                    if symbol == 'Q':
                        symbol = 'X'
                    position = tuple([Bh_TO_ANG * float(x)
                                     for x in fields[0:3]])
                except ValueError as e:
                    raise e
            elif len(fields) == 3:  # gradients
                #  -.51654903354681D-07  -.51654903206651D-07  0.51654903169644D-07
                grad = []
                for val in fields[:3]:
                    try:
                        grad.append(
                            -float(val.replace('D', 'E')) *
                            AU_TO_EV / Bh_TO_ANG
                        )
                    except ValueError as e:
                        raise e
            else:  # yield and to next cycle
                info_dict = dict()
                atom_names, info_dict['atom_types'], atom_numbs = np.unique(
                    symbol, return_inverse=True, return_counts=True)
                info_dict['atom_names'] = list(atom_names)
                info_dict['atom_numbs'] = list(atom_numbs)
                info_dict['forces'] = np.array(grad)
                info_dict['energies'] = np.array(energy)
                info_dict['cells'] = np.array(
                    [[[100., 0., 0.], [0., 100., 0.], [0., 0., 100.]]])

                yield info_dict
                break

from dpdata.format import Format


@Format.register("xtb/gradient")
class XTBGradientFormat(Format):
    def from_labeled_system(self, grad_file_name, **kwargs):
        grad_file_obj = open(grad_file_name, "r")
        data = list(dpdata.xtb.xtb_xyz.get_grad_frame(grad_file_obj))[0]
        return data

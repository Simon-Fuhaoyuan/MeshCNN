import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh

class ReconstructionData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.subroot = opt.recon_subroot
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_target_dataset(self.dir, self.subroot, opt.phase)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta = {'mesh': mesh, 'label': edge_features}
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_target_dataset(dir, subroot, phase):
        meshes = []
        if subroot not in os.listdir(dir):
            print('Error: No class %s in %s' % (subroot, dir))
            exit()
        dir = os.path.join(dir, subroot)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_mesh_file(fname) and (root.count(phase)==1):
                    path = os.path.join(root, fname)
                    meshes.append(path)
        return meshes

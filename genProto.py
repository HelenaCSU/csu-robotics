from caffe.proto import caffe_pb2
import shutil
import os
import math


class GenProtoText(object):
    def __init__(self):
        self.train_net = ["/home/csu/Downloads/py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/stage1_rpn_train.pt",
                          "/home/csu/Downloads/py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/stage1_fast_rcnn_train.pt",
                          "/home/csu/Downloads/py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/stage1_rpn_train.pt",
                          "/home/csu/Downloads/py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/stage1_fast_rcnn_train.pt"]
        self.lr = 1e-5
        self.lr_policy = 'step'
        self.display = 20
        self.average_loss = 100
        self.momentum = 9e-1
        self.snapshot = 0
        self.gamma = 1e-1
        self.snapshot_prefix = ["zf_rpn", "zf_fast_rcnn", "zf_rpn", "zf_fast_rcnn"]
        self.file_name=['../models/pascal_voc/ZF/faster_rcnn_alt_opt/stage1_rpn_solver60k80k.pt',
                        '../models/pascal_voc/ZF/faster_rcnn_alt_opt/stage1_fast_rcnn_solver30k40k.pt',
                        '../models/pascal_voc/ZF/faster_rcnn_alt_opt/stage2_rpn_solver60k80k.pt',
                        '../models/pascal_voc/ZF/faster_rcnn_alt_opt/stage2_fast_rcnn_solver30k40k.pt']


    # def floatRange(start, stop, step):
    #     return [start + float(i) * (stop - start) / (float(step) - 1) for i in range(step)]


    def Generate(self, stepsize, weight_dacay):
        for i in range(4):
            # for stepsize in range(300,600,100):
            #     for weight_dacay in range():
                    s = caffe_pb2.SolverParameter()
                    s.train_net = self.train_net[i]
                    s.base_lr = self.lr
                    s.lr_policy = self.lr_policy
                    s.stepsize = stepsize
                    s.display = self.display
                    s.gamma = self.gamma
                    s.average_loss = self.average_loss
                    s.momentum = self.momentum
                    s.weight_decay = weight_dacay
                    s.snapshot = self.snapshot
                    s.snapshot_prefix = self.snapshot_prefix[i]

                    with open(self.file_name[i], 'w') as f:
                        f.write(str(s))

    # def generateList(stepmin,stepmax,stepst,wdmin,wdmax,wdst):
    #
    #     L=[ (i,j/1000) for i in range(stepmin,stepmax,stepst) for j in range(wdmin,wdmax,wdst) ]
    #     return L


if __name__ == "__main__":
    gen = GenProtoText()
    L=gen.generateList(2000,3000,1000,1,10,1)
    print(L)

    # for stepsize in range(3000,5000,500)
    #     for weight_decay in range()
    # gen.Generate(300, 0.0005)

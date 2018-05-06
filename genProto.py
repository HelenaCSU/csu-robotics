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

if __name__ == "__main__":
    gen = GenProtoText()
    # 清除每次生成的cache
    output_path = ['/home/csu/Downloads/py-faster-rcnn/output']
    cache_path = ['/home/csu/Downloads/py-faster-rcnn/data/cache']
    annotations_cache = ["/home/csu/Downloads/py-faster-rcnn/data/VOCdevkit2007/annotations_cache"]
    shutil.rmtree(output_path[0])
    shutil.rmtree(cache_path[0])
    shutil.rmtree(annotations_cache[0])
    # 调用generate函数
    # for stepsize in range(3000,5000,500)
    #     for weight_decay in range()
    # gen.Generate(300, 0.0005)
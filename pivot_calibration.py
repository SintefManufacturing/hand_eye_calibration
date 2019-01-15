# coding=utf-8

"""
A standard tool-tip calibration procedure found in most robot controllers is based on pivot calibration. Assume a tool tip point Ft fixed in the tool flange, F. Sample a series of flange poses in base coordinates, BFi, where t is kept also fixed with respect to the robot base frame, B, such that Bt is also a fixed point. Then for all i,j we have BFi * Ft = Bt = BFj * Ft. Formulating BFi = [Ri, ti] and casting the equation as Ax=b we can obtain a stacked system of equations for some set of indices {(i,j)}_i!=j:
[Ri - Rj]      [tj - ti]
[  ...  ] Ft = [  ...  ]
[  ...  ]      [  ...  ]
(Note the order of indices on the opposing sides!)

We can isolate for Ft, and then estimate Bt as the average over {BFi * Ft}_i.

This method is described as "Algebraic Two Step" in the publication

@inproceedings{yaniv2015pivot,
  title={Which pivot calibration?},
  author={Yaniv, Ziv},
  booktitle={Medical Imaging 2015: Image-Guided Procedures, Robotic Interventions, and Modeling},
  volume={9415},
  pages={941527},
  year={2015},
  organization={International Society for Optics and Photonics}
}

"""

__author__ = "Morten Lind"
__copyright__ = "SINTEF 2019"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten.lind@sintef.no"
__status__ = "Development"


from itertools import combinations

import numpy as np
import math3d as m3d


class PivotCalibrator:

    def __init__(self, base_flange_s):
        self._bfs = base_flange_s
        self._combs = list(combinations(self._bfs, 2))
        # print(list(self._combs))

    def __call__(self):
        A = np.vstack([bfi.orient.array - bfj.orient.array
                       for (bfi, bfj) in self._combs])
        # print(A)
        # print(self._combs[0][0].pos)
        dbfps = [(bfj.pos.array - bfi.pos.array).reshape((3,1))
                 for (bfi, bfj) in self._combs]
        b = np.vstack(dbfps)
        ft = np.linalg.pinv(A).dot(b).reshape(-1)
        # print(ft)
        self.ft = m3d.Vector(ft)
        # print(self.ft)
        bt = np.average([(bfi * self.ft).array.reshape(-1)for bfi in self._bfs], axis=0)
        # print(bt)
        self.bt = m3d.Vector(bt)
        return self.ft, self.bt


def _test(n=5, noise=0.005):
    """Test and return the identification errors for a generated set of
    poses and applying a given noise to the synthetic ft when
    generating poses.
    """
    bt = m3d.Vector(0.5, 0.6, 0.7)
    ft = m3d.Vector(0.1, 0.05, 0.15)
    # Generate random orientation Assume limited ergodicity
    bfs = []
    for i in range(n):
        o = m3d.Orientation.new_euler(np.random.uniform(0, 1, 3))
        p = bt - o * (ft + m3d.Vector(np.random.uniform(-noise, noise, 3)))
        bfs.append(m3d.Transform(o,p))
    pc = PivotCalibrator(bfs)
    pc()
    return (pc.ft - ft).length, (pc.bt - bt).length
    

def _test_identify(range_=(3,10), rep=100, noise=0.005):
    """Plot the error in identifying ft for a given range of poses, with a
    number of repetitions for every pose number and a given noise.
    """
    import matplotlib.pyplot as plt
    err = []
    for n in range(*range_):
        for i in range(rep):
            err.append((n, _test(n, noise)[0]))
    plt.scatter(*np.array(err).T, marker='+')
    plt.title('Ft error, noise={}'.format(noise))
    plt.show()

from collections import namedtuple
from gen.trees import OpDesc

M = "M"
K = "K"
N = "N"
B = "B"

def _make_op(name, inputs, output):
  return OpDesc(name, tuple(map(tuple, inputs)), tuple(output))

ops = [
  # _make_op("elementwise0", [[],[]], []),
  # _make_op("elementwise1", [[M],[M]], [M]),
  # _make_op("elementwise2", [[M, N], [M, N]], [M, N]),
  _make_op("matmul", [[M, K], [K, N]], [M, N]),
  # _make_op("batch_matmul", [[B, M, K],  [B, K, N]],  [B, M, N]),
  _make_op("matvec", [[M, N], [N]], [M]),
  # _make_op("vecmat", [[M], [M, N]], [N]),
  # _make_op("dot", [[M], [M]], []),
]

ops = dict((op.name, op) for op in ops)

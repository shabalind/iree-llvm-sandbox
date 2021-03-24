import mlir.ir as ir
import mlir.execution_engine as ee
import mlir.dialects.linalg.opdsl.lang as tc


T1 = tc.TV.T1
T2 = tc.TV.T2
S = tc.S
U = tc.U
D = tc.D


@tc.linalg_structured_op
def matmul(A=tc.TensorDef(T1, S.M, S.K),
           B=tc.TensorDef(T2, S.K, S.N),
           C=tc.TensorDef(U, S.M, S.N, output=True)):
  """Performs a matrix multiplacation of two 2D inputs.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  tc.implements(tc.ContractionOpInterface)
  C[D.m, D.n] += tc.cast(U, A[D.m, D.k]) * tc.cast(U, B[D.k, D.n])


def invoke(op):
  pass

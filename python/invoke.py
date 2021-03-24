import ctypes
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


def to_mlir(op):
  return r"""
func @entrypoint() -> f64 attributes { llvm.emit_c_interface } {
  %t_start = call @rtclock() : () -> f64
  %t_end = call @rtclock() : () -> f64
  %t_delta = subf %t_end, %t_start: f64
  return %t_delta : f64
}
func @rtclock() -> f64 {
  %zero = constant 0.0 : f64
  return %zero : f64
}
  """


def to_llvm(module):
  import mlir.conversions
  import mlir.passmanager as pm
  pm = pm.PassManager.parse("convert-std-to-llvm")
  pm.run(module)
  print(module)
  return module


def invoke(op):
  with ir.Context():
    module = ir.Module.parse(to_mlir(op))
    engine = ee.ExecutionEngine(to_llvm(module))
    c_double_p = ctypes.c_double * 1
    res = c_double_p(-1.)
    engine.invoke("entrypoint", res)
    return res


if __name__ == "__main__":
  invoke(matmul)

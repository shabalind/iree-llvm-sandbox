"""Utilities for search space exploration over linalg operations."""

from mlir.ir import *
from mlir.dialects import linalg, std
from mlir.dialects.linalg.opdsl.lang import TensorDef, LinalgOpDef, S, TV
from itertools import chain
from random import choice, randrange
from compilation import f16, f32, f64, scalar_types, compile_and_callback
from transforms import expert_compilerr_1
from gen.gen import generate_programs
from gen.trees import Program
from gen.util import dump_expr
from search import collect_variables, instantiate_variables


ops = [linalg.matmul, linalg.matvec, linalg.vecmat, linalg.dot]
name_to_op = dict((op.op_name, op) for op in ops)


def to_linalg_model(program: Program):
  model = LinalgOpDef("composite_op")
  dim_syms = [getattr(S, "D" + str(i)) for (i, _) in enumerate(program.dims)]
  dim_to_sym = dict(zip(program.dims, dim_syms))

  for param in program.variables:
    dims = [dim_to_sym[dim] for dim in param.dims]
    tdef = TensorDef(TV.T, *dims)
    model.add_tensor(param.name, tdef)

  result_dims = [dim_to_sym[dim] for dim in program.result_dims()]
  result_tdef = TensorDef(TV.T, *dims, output=True)
  model.add_tensor("result", result_tdef)

  return model


def to_linalg_op(program: Program):
  model = to_linalg_model(program)

  def composite_op(*ins, outs):
    assert(len(ins) == len(program.variables))

    expr_to_value = dict(zip(map(id, program.variables), ins))
    op_count = len(program.ops)
    outputs = dict()
    element_type = F32Type.get()

    for (i, expr) in enumerate(program.ops):
      op = name_to_op[expr.op_name]

      if i != op_count - 1:
        zero = std.ConstantOp(
          value=FloatAttr.get(element_type, 0.),
          result=element_type).result
        sizes = [128
                 for _ in op.model.outputs[0].shape]
        init_tensor_op = linalg.InitTensorOp(sizes, element_type)
        init_tensor = init_tensor_op.results[0]
        fill_tensor = linalg.FillOp(output=init_tensor, value=zero).results[0]
        outputs[id(expr)] = fill_tensor
      else:
        outputs[id(expr)] = outs[0]

    for expr in program.ops:
      op = name_to_op[expr.op_name]
      inargs = [expr_to_value[id(arg)] for arg in expr.args]
      outarg = outputs[id(expr)]
      value = op(*inargs, outs=[outarg], emit_generic=True)
      expr_to_value[id(expr)] = value

    return expr_to_value[id(program.result)]

  composite_op.model = model

  return composite_op


def generate_ops(size: int):
  for program in generate_programs(size):
    print('---')
    dump_expr(program.result)
    if len(program.ops) > 0:
      yield to_linalg_op(program)


def main():
  for op in generate_ops(10):
    variables = collect_variables(op, [f32], range(128, 128+1))
    assignments = instantiate_variables(variables)
    compile_and_callback(op, expert_compilerr_1, lambda x: None, **assignments)


if __name__ == "__main__":
  main()

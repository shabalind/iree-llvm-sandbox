import ctypes
import mlir.ir as ir
import mlir.execution_engine as ee
import mlir.dialects.linalg.opdsl.lang as tc
import mlir.conversions
import mlir.passmanager as pm


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


def to_mlir(op, input_shapes, output_shapes, iterations):
  output = []
  fresh_counter = 0
  variables = dict()

  def assign_var(var, value):
    nonlocal variables
    if var not in variables:
      variables[var] = value
    else:
      assert variables[var] == value

  def assign_variables(tdefs, shapes):
    for (tdef, (ty, dims)) in zip(tdefs, shapes):
      assign_var(tdef.type_var, ty)
      for (sym, dim) in zip(tdef.shape, dims):
        assign_var(sym, dim)

  def fresh():
    nonlocal fresh_counter
    res = "%v" + str(fresh_counter)
    fresh_counter += 1
    return res

  def tensor_type(tdef):
    parts = []
    for sym in tdef.shape:
      parts.append(str(variables[sym]))
    parts.append(variables[tdef.type_var])
    return "tensor<{}>".format("x".join(parts))

  def emit_func_start(name, arg_types=[], ret_type=None):
    arg_names = []
    args = []
    for arg_ty in arg_types:
      arg_name = fresh()
      arg_names.append(arg_name)
      args.append("{}: {}".format(arg_name, arg_ty))
    ret = "" if ret_type is None else " -> {}".format(ret_type)
    output.append(r"func @{}({}) {} {{".format(name, ", ".join(args), ret))
    return arg_names

  def emit_func_end():
    output.append(r"}")

  def emit_return(res=None):
    output.append("return" if res is None else "return " + str(res))

  def emit_constant(value, ty):
    name = fresh()
    output.append("{} = constant {} : {}".format(name, value, ty))
    return name

  # Initialize tensor of given type and return a fresh name for the result.
  # All variables and symbols should be bound beforehand.
  def emit_init_tensor(tdef, value):
    init_name = fresh()
    fill_name = fresh()
    scalar_ty = variables[tdef.type_var] 
    tensor_ty = tensor_type(tdef)
    dims = [variables[sym] for sym in tdef.shape]
    output.append("{} = linalg.init_tensor {} : {}".format(
        init_name, dims, tensor_ty))
    output.append("{} = linalg.fill({}, {}) : {}, {} -> {}".format(
        fill_name, init_name, value, tensor_ty, scalar_ty, tensor_ty))
    return fill_name


  def emit_for_start(init, to, step, arg_init, arg_type):
    name = fresh()
    i = fresh()
    arg = fresh()
    parts = []
    parts.append("{} = scf.for".format(name))
    parts.append("{} = {} to {} step {}".format(i, init, to, step))
    parts.append("iter_args({} = {}) -> {}".format(arg, arg_init, arg_type))
    parts.append("{") 
    output.append(" ".join(parts))
    return name

  def emit_for_end():
    output.append("}")

  def emit_yield(value, ty):
    output.append("scf.yield {} : {}".format(value, ty))

  def emit_call(func_name, args, arg_types, ret_type):
    name = fresh()
    output.append("{} = call @{}({}): ({}) -> {}".format(
        name, func_name, ", ".join(args), ", ".join(arg_types), ret_type))
    return name

  # Generate a function that invokes the op once.
  # Currently it doesn't actually do anythign but return
  # a tensor of the return type.
  def emit_op_func():
    assert(len(op.model.outputs) == 1)
    output = op.model.outputs[0]
    output_type = tensor_type(output)
    arg_types = [tensor_type(i) for i in op.model.inputs]
    arg_types.append(output_type)
    args = emit_func_start("op", arg_types, output_type)
    value = emit_constant("0.0", variables[output.type_var])
    res = emit_init_tensor(output, value)
    emit_return("{}: {}".format(res, output_type))
    emit_func_end()

  # Entry-point invoked by the execution engine.
  # Initialializes input and output tensors and invokes
  # the op repeatedly on them.
  def emit_entry_func():
    emit_func_start("entry")

    # Initialize input tensors.
    input_values = []
    input_types = []
    for inp in op.model.inputs:
      input_scalar = variables[inp.type_var]
      input_constant = emit_constant("0.0", input_scalar)
      input_values.append(emit_init_tensor(inp, input_constant))
      input_types.append(tensor_type(inp))

    # Initialize a single output tensor.
    assert(len(op.model.outputs) == 1)
    output = op.model.outputs[0]
    output_scalar = variables[output.type_var]
    output_constant = emit_constant("1.0", output_scalar)
    output_value = emit_init_tensor(output, output_constant)
    output_type = tensor_type(output)

    # Start of the loop.
    loop_init = emit_constant("0", "index")
    loop_to = emit_constant("1000", "index")
    loop_step = emit_constant("1", "index")
    emit_for_start(loop_init, loop_to, loop_step, output_value, output_type) 

    # A single call to the op function within the loop body.
    call_args = []
    call_args.extend(input_values)
    call_args.append(output_value)
    call_arg_types = []
    call_arg_types.extend(input_types)
    call_arg_types.append(output_type)
    call_result = emit_call("op", call_args, call_arg_types, output_type) 

    # End of the loop.
    emit_yield(call_result, output_type)
    emit_for_end()

    # End of the entry function.
    emit_return()
    emit_func_end()

  assign_variables(op.model.inputs, input_shapes)
  assign_variables(op.model.outputs, output_shapes)
  emit_op_func()
  emit_entry_func()

  return "\n".join(output)


def to_llvm(module):
  manager = pm.PassManager.parse("convert-std-to-llvm")
  manager.run(module)
  return module


def invoke(op, input_shapes, output_shapes, iterations):
  with ir.Context():
    mlir_text = to_mlir(op, input_shapes, output_shapes, iterations)
    print("-- TEXT:")
    print(mlir_text)
    mlir_module = ir.Module.parse(mlir_text)
    print("-- MLIR:")
    print(mlir_module)
    llvm_module = to_llvm(mlir_module)
    print("-- LLVM:")
    print(llvm_module)
    # engine = ee.ExecutionEngine(llvm_module)
    # engine.invoke("entry")


if __name__ == "__main__":
  input_shapes = [
    ("f32", [100, 10]),
    ("f32", [10, 20]),
  ]
  output_shapes = [
    ("f32", [100, 20]),
  ]
  invoke(matmul, input_shapes, output_shapes, iterations=1000)

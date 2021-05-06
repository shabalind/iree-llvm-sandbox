from collections import defaultdict
from itertools import product
from typing import Iterator, List, Dict
from gen.ops import ops
from gen.trees import *
from gen.util import dump_expr, FreshNames
from gen.unify import unify_program


def generate_exprs(op_descs: List[OpDesc], var_ranks: Dict[VarName, int],
                   expr_size: int, fresh_names: FreshNames) -> List[Expr]:
  """Generate all expressions of a given size.

  Params:
    op_descs: Descriptions of all the operations to be used.
    var_ranks: The names and tensor ranks of variables.
    expr_size: The size of expressions to generate.
    fresh_names: An instance of FreshNames to generate fresh dimension names.

  Returns:
    A list of generated expressions.
  """

  def arg_size_choices(n: int, k: int, result=[]):
    """Return all n-element lists `cs = [c1, ..., cn]` such that `sum(cs) == k` and each `ci > 0`."""
    if n == 0:
      yield result
    elif n == 1:
      result.append(k)
      yield result
      result.pop()
    else:
      for c in range(1, k - n + 2):
        result.append(c)
        yield from arg_size_choices(n - 1, k - c, result)
        result.pop()

  # Dictionary containing all expressions generated so far. Indexed by rank of
  # output tensor (first), and size (second).
  memo = defaultdict(
      lambda: defaultdict(list))  # : Dict[int, Dict[int, List[Expr]]]

  # Populate memo with all expressions of size 1.
  # TODO(gsps): Also add expressions for all constant ops in `op_descs`.
  for var_name, var_rank in var_ranks.items():
    fresh_dim_args = [fresh_names.fresh(f'{var_name}') for i in range(var_rank)]
    memo[var_rank][1].append(VarExpr(var_name, tuple(fresh_dim_args)))

  for k in range(2, expr_size + 1):
    # To generate expression of size `k`, pick the op, then pick arguments whose
    # sizes sum to `k-1`.
    for op_desc in op_descs:
      arg_ranks = [len(in_shape) for in_shape in op_desc.in_shapes]

      # For every combination of arg sizes that sums to `k - 1`:
      num_args = len(op_desc.in_shapes)
      for arg_sizes in arg_size_choices(num_args, k - 1):

        # For every combination of arguments of compatible sizes and ranks:
        arg_choices = product(
            *(memo[arg_rank][arg_size]
              for arg_rank, arg_size in zip(arg_ranks, arg_sizes)))
        for args in arg_choices:
          # Collect, deduplicate and freshen the dimension arguments.
          fresh_dim_args = [
              fresh_names.fresh(dim_arg) for dim_arg in op_desc.dim_args()
          ]
          # Create the new op and store it in the memo.
          op = OpExpr(op_desc.name, tuple(fresh_dim_args), tuple(args))
          memo[len(op_desc.out_shape)][k].append(op)

  # Extract all the expressions of the desired size from the memo.
  return sum((memo_by_rank[size]
              for size in range(expr_size + 1)
              for memo_by_rank in memo.values()), [])


def generate_programs(expr_size: int) -> Iterator[Program]:
  fresh_names = FreshNames()

  var_ranks = {
      'x1': 1,
      'y1': 1,
      'z1': 1,
      'x2': 2,
      'y2': 2,
      'z2': 2,
  }

  print(f'Generating all expressions of size = {expr_size} ...')
  exprs = generate_exprs(ops.values(), var_ranks, expr_size, fresh_names)
  for expr in exprs:
    yield unify_program(expr)


if __name__ == '__main__':
  expr_size = 3

  print(f'Generating all expressions of size = {expr_size} ...')

  program_count = 0

  for program in generate_programs(expr_size):
    nops = len(program.ops)
    nvars = len(program.variables)
    ndims = len(program.dims)
    print(f'--- ops={nops}, vars={nvars}, dims={ndims}')
    dump_expr(program.result)
    program_count = program_count + 1

  print(f'Generated {program_count} expressions.')

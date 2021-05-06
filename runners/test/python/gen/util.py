from collections import defaultdict
from typing import List, DefaultDict, Dict
from gen.trees import *


class FreshNames(object):
  """Generate fresh names."""
  next: DefaultDict[str, int]

  def __init__(self):
    self.next = defaultdict(lambda: 0)

  def fresh(self, name: str) -> str:
    n = self.next[name]
    self.next[name] = n + 1
    return f'{name}__{n}'


def topological_sort(expr: Expr) -> List[Expr]:
  seen = set()
  output = []

  def visit(expr: Expr):
    nonlocal seen, output
    if expr not in seen:
      seen.add(expr)
      if isinstance(expr, VarExpr):
        pass
      elif isinstance(expr, OpExpr):
        for arg in expr.args:
          visit(arg)
      else:
        raise Exception("unknown expr: " + str(expr))
      output.append(expr)

  visit(expr)
  return output


def dump_expr(expr: Expr):
  schedule = topological_sort(expr)
  op_num_by_expr = dict((expr, i) for i, expr in enumerate(schedule))

  op_num = 0
  for expr in schedule:
    if isinstance(expr, VarExpr):
      dims_str = ', '.join(expr.dims)
      print(f'%{op_num} = {expr.name} : [{dims_str}]')
    elif isinstance(expr, OpExpr):
      dim_args_str = ', '.join(expr.dim_args)
      args_str = ', '.join(f'%{op_num_by_expr[arg]}' for arg in expr.args)
      print(f'%{op_num} = {expr.op_name}<{dim_args_str}>({args_str})')
    else:
      raise Exception('unknown expr: ' + str(expr))
    op_num += 1

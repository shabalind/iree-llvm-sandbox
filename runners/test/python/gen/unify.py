from typing import List, Set
from pprint import pprint
from gen.trees import *
from gen.ops import ops
from gen.util import FreshNames, topological_sort


def subst_desc(dim_args: Tuple[DimName], desc: OpDesc) -> OpDesc:
  subst_dict = dict((from_dim, to_dim)
                    for (from_dim, to_dim)
                    in zip(desc.dim_args(), dim_args))
  subst = lambda from_dim: subst_dict[from_dim]
  in_shapes = tuple(tuple(map(subst, in_shape))
                    for in_shape in desc.in_shapes)
  out_shape = tuple(map(subst, desc.out_shape))
  return OpDesc(desc.name, in_shapes, out_shape)


def collect_constraints(sorted_expr: List[Expr]) -> DimConstr:
  constraints = list() # : DimConstr
  env = dict() # : Dict[ExprId, Expr]

  for expr in sorted_expr:
    if isinstance(expr, VarExpr):
      env[id(expr)] = expr.dims
    elif isinstance(expr, OpExpr):
      op_desc = ops[expr.op_name]
      new_desc = subst_desc(expr.dim_args, op_desc)
      env[id(expr)] = new_desc.out_shape
      for (shape, arg) in zip(new_desc.in_shapes, expr.args):
        for (l_dim, r_dim) in zip(shape, env[id(arg)]):
          constraints.append((l_dim, r_dim))
    else:
      raise Exception("unknown expr: " + str(expr))

  return constraints


def unify_equiv_classes(constraints: DimConstr) -> Dict[DimName, int]:
  all_vars = sorted(set(v
                        for p in constraints
                        for v in p))
  eq_classes = dict((i, i) for (i, _) in enumerate(all_vars))
  var_to_class = dict((v, i) for (i, v) in enumerate(all_vars))
  eq_class_ranks = dict((i, 1) for (i, _) in enumerate(all_vars))

  def get_class(dim_var):
    nonlocal var_to_class, eq_classes
    root_class = var_to_class[dim_var]
    while True:
      parent_class = eq_classes[root_class]
      if parent_class == root_class:
        break
      # Do path splitting to shorten paths.
      eq_classes[root_class] = eq_classes[parent_class]
      root_class = parent_class;
    return root_class

  for constr in constraints:
    (l, r) = constr
    l_class = get_class(l)
    r_class = get_class(r)
    if l_class == r_class:
      # already unified
      pass
    else:
      # merge two classes by settting one as the root for the other
      l_rank, r_rank = eq_class_ranks[l_class], eq_class_ranks[r_class]
      if l_rank < r_rank:
        eq_classes[l_class] = r_class
      else:
        eq_classes[r_class] = l_class
        if l_rank == r_rank:
          eq_class_ranks[l_class] += 1

  return dict((v, get_class(v)) for v in all_vars)


def unify_program(expr: Expr) -> Program:
  sorted_expr = topological_sort(expr)
  constraints = collect_constraints(sorted_expr)
  equiv_classes = unify_equiv_classes(constraints)

  fresh_names = FreshNames()
  free_names = dict() # : Dict[DimName, DimName]
  class_names = dict() # : Dict[int, DimName]
  transformed_expr = dict() # : Dict[ExprId, Expr]
  dims = set()

  def transform_dim(dim: DimName) -> DimName:
    nonlocal equiv_classes, equiv_classes, fresh_names, free_names
    if dim in equiv_classes:
      cls = equiv_classes[dim]
      if cls not in class_names:
        class_names[cls] = fresh_names.fresh("D")
      return class_names[cls]
    else:
      if dim not in free_names:
        free_names[dim] = fresh_names.fresh("D")
      return free_names[dim]

  def transform_dims(dims: Tuple[DimName]) -> Tuple[DimName]:
    return tuple(map(transform_dim, dims))

  variables = []
  ops = []

  for expr in sorted_expr:
    if isinstance(expr, VarExpr):
      new_expr = VarExpr(expr.name, transform_dims(expr.dims))
      transformed_expr[id(expr)] = new_expr
      variables.append(new_expr)
    elif isinstance(expr, OpExpr):
      new_dims = transform_dims(expr.dim_args)
      new_args = tuple(transformed_expr[id(arg)] for arg in expr.args)
      new_expr = OpExpr(expr.op_name, new_dims, new_args)
      transformed_expr[id(expr)] = new_expr
      ops.append(new_expr)
    else:
      raise Exception("unknown expr: " + str(expr))

  dims = sorted(list(free_names.values()) + list(class_names.values()))
  body = transformed_expr[id(expr)]
  return Program(dims, tuple(variables), tuple(ops), body)


if __name__ == "__main__":
  v1 = VarExpr("v1", ("D1",))
  v2 = VarExpr("v2", ("D2",))
  v3 = OpExpr("dot", ("D3",), (v1, v2))
  pprint(unify_expr(v3))

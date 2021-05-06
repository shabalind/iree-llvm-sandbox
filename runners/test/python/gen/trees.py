from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

# S := [d1 x ... x dn]
#
# t := x | op<d1,...,dm>(t1,...,tn)
#
# G := <> | G,x:S
#
# P := d1...dm; G |- t
#
# size t ... size of the term
# size x = 1
# size op(t1,...,tn) = 1 + Sum_i size(ti)

DimName = str
VarName = str
Shape = Tuple[DimName]
DimConstr = List[Tuple[DimName, DimName]]
DimSubst = Dict[DimName, DimName]


@dataclass
class OpDesc:
  """An abstract description of an operation (i.e. the generic signature)."""
  name: str
  in_shapes: Tuple[Shape]
  out_shape: Shape

  def dim_args(self) -> List[DimName]:
    """Infer the list of dimension arguments from the input and output shapes."""
    dim_args = set(self.out_shape)
    for in_shape in self.in_shapes:
      dim_args.update(in_shape)
    return sorted(dim_args)


@dataclass(frozen=True)
class VarExpr:
  name: VarName
  dims: Tuple[DimName]


@dataclass(frozen=True)
class OpExpr:
  op_name: str
  dim_args: Tuple[DimName]
  args: Tuple['Expr']


Expr = Union[VarExpr, OpExpr]


@dataclass(frozen=True)
class Program:
  dims: Tuple[DimName]
  variables: Tuple[VarExpr]
  ops: Tuple[OpExpr]
  result: Expr

  def result_dims(self) -> Tuple[DimName]:
    if isinstance(self.result, VarExpr):
      return self.result.dims
    else:
      from gen.ops import ops
      from gen.unify import subst_desc
      desc = ops[self.result.op_name]
      subst = subst_desc(self.result.dim_args, desc)
      return subst.out_shape

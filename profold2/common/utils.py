from __future__ import annotations

import functools
from dataclasses import dataclass
import re

from Bio.PDB import Model


class ExpressionTreeEvaluator:
  """A class for evaluating custom logical parenthetical expressions. The
     implementation is very generic, supports nullary, unary, and binary
     operators, and does not know anything about what the expressions actually
     mean. Instead the class interprets the expression as a tree of sub-
     expressions, governed by parentheses and operators, and traverses the
     calling upon a user-specified evaluation function to evaluate leaf
     nodes as the tree is gradually collapsed into a single node. This
     can be used for evaluating set expressions, algebraic expressions, and
     others.

  Args:
    operators_nullary (list): A list of strings designating nullary operators
        (i.e., operators that do not have any operands). E.g., if the language
        describes selection algebra, these could be "hyd", "all", or "none"].
    operators_unary (list): A list of strings designating unary operators
        (i.e., operators that have one operand, which must comes to the right
        of the operator). E.g., if the language describes selection algebra,
        these could be "name", "resid", or "chain".
    operators_binary (list): A list of strings designating binary operators
        (i.e., operators that have two operands, one on each side of the
        operator). E.g., if the language describes selection algebra, thse
        could be "and", "or", or "around".
    eval_function (str): A function that is able to evaluate a leaf node of
        the expression tree. It shall accept three parameters:

        operator (str): name of the operator
        left: the left operand. Will be None if the left operand is missing or
            not relevant. Otherwise, can be either a list of strings, which
            should represent an evaluatable sub-expression corresponding to the
            left operand, or the result of a prior evaluation of this function.
        right: Same as `left` but for the right operand.

        The function should attempt to evaluate the resulting expression and
        return None in the case of failing or a dictionary with the result of
        the evaluation stored under key "result".
    left_associativity (bool): If True (the default), operators are taken to be
        left-associative. Meaning something like "A and B or C" is "(A and B) or C".
        If False, the operators are taken to be right-associative, such that
        the same expression becomes "A and (B or C)". NOTE: MST is right-associative
        but often human intiution tends to be left-associative.
    debug (bool): If True (default is false), will print a great deal of debugging
        messages to help diagnose any evaluation problems.
  """

  def __init__(
      self,
      operators_nullary: list,
      operators_unary: list,
      operators_binary: list,
      eval_function: function,
      left_associativity: bool = True,
      debug: bool = False,
  ):
    self.operators_nullary = operators_nullary
    self.operators_unary = operators_unary
    self.operators_binary = operators_binary
    self.operators = operators_nullary + operators_unary + operators_binary
    self.eval_function = eval_function
    self.debug = debug
    self.left_associativity = left_associativity

  def _traverse_expression_tree(self, E, i=0, eval_all=True, debug=False):
    def _collect_operands(E, j):
      # collect all operands before hitting an operator
      operands = []
      for k in range(len(E[j:])):
          if E[j + k] in self.operators:
              k = k - 1
              break
          operands.append(E[j + k])
      return operands, j + k + 1

    def _find_matching_close_paren(E, beg: int):
      c = 0
      for i in range(beg, len(E)):
          if E[i] == "(":
              c = c + 1
          elif E[i] == ")":
              c = c - 1
          if c == 0:
              return i
      return None

    def _my_eval(op, left, right, debug=False):
      if debug:
          print(
              f"\t-> evaluating {operand_str(left)} | {op} | {operand_str(right)}"
          )
      result = self.eval_function(op, left, right)
      if debug:
          print(f"\t-> got result {operand_str(result)}")
      return result

    def operand_str(operand):
      if isinstance(operand, dict):
          if "result" in operand and len(operand["result"]) > 15:
              vec = list(operand["result"])
              beg = ", ".join([str(i) for i in vec[:5]])
              end = ", ".join([str(i) for i in vec[-5:]])
              return "{'result': " + f"{beg} ... {end} ({len(vec)} long)" + "}"
          return str(operand)
      return str(operand)

    left, right, op = None, None, None
    if debug:
      print(f"-> received {E[i:]}")

    while i < len(E):
      if all([x is None for x in (left, right, op)]):
          # first part can either be a left parenthesis, a left operand, a nullary operator, or a unary operator
          if E[i] == "(":
              end = _find_matching_close_paren(E, i)
              if end is None:
                  return None, f"parenthesis imbalance starting with {E[i:]}"
              # evaluate expression inside the parentheses, and it becomes the left operand
              left, rem = self._traverse_expression_tree(
                  E[i + 1 : end], 0, eval_all=True, debug=debug
              )
              if left is None:
                  return None, rem
              i = end + 1
              if not eval_all:
                  return left, i
          elif E[i] in self.operators_nullary:
              # evaluate nullary op
              left = _my_eval(E[i], None, None, debug)
              if left is None:
                  return None, f"failed to evaluate nullary operator '{E[i]}'"
              i = i + 1
          elif E[i] in self.operators_unary:
              op = E[i]
              i = i + 1
          elif E[i] in self.operators:
              # an operator other than a unary operator cannot appear first
              return None, f"unexpected binary operator in the context {E[i:]}"
          else:
              # if not an operator, then we are looking at operand(s)
              left, i = _collect_operands(E, i)
      elif (left is not None) and (op is None) and (right is None):
          # we have a left operand and now looking for a binary operator
          if E[i] not in self.operators_binary:
              return (
                  None,
                  f"expected end or a binary operator when got '{E[i]}' in expression: {E}",
              )
          op = E[i]
          i = i + 1
      elif (
          (left is None) and (op in self.operators_unary) and (right is None)
      ) or (
          (left is not None) and (op in self.operators_binary) and (right is None)
      ):
          # we saw a unary operator before and now looking for a right operand, another unary operator, or a nullary operator
          # OR
          # we have a left operand and a binary operator before, now looking for a right operand, a unary operator, or a nullary operator
          if (
              E[i] in (self.operators_nullary + self.operators_unary)
              or E[i] == "("
          ):
              right, i = self._traverse_expression_tree(
                  E, i, eval_all=not self.left_associativity, debug=debug
              )
              if right is None:
                  return None, i
          else:
              right, i = _collect_operands(E, i)

          # We are now ready to evaluate, because:
          #   we have a unary operator and a right operand
          #   OR
          #   we have a left operand, a binary operator, and a right operand
          result = _my_eval(op, left, right, debug)
          if result is None:
              return (
                  None,
                  f"failed to evaluate operator '{op}' (in expression {E}) with operands {operand_str(left)} and {operand_str(right)}",
              )
          if not eval_all:
              return result, i
          left = result
          op, right = None, None

      else:
          return (
              None,
              f"encountered an unexpected condition when evaluating {E}: left is {operand_str(left)}, op is {op}, or right {operand_str(right)}",
          )

    if (op is not None) or (right is not None):
      return None, f"expression ended unexpectedly"
    if left is None:
      return None, f"failed to evaluate expression: {E}"

    return left, i

  def evaluate(self, expression: str):
    """Evaluates the expression and returns the result."""

    def _split_tokens(expr):
      # first split by parentheses (preserving the parentheses themselves)
      parts = list(re.split("([()])", expr))
      # then split by space (getting rid of space)
      return [
          t.strip()
          for p in parts
          for t in re.split("\s+", p.strip())
          if t.strip() != ""
      ]

    # parse expression into tokens
    E = _split_tokens(expression)
    val, rem = self._traverse_expression_tree(E, debug=self.debug)
    if val is None:
      raise Exception(
          f"failed to evaluate expression: '{expression}', reason: {rem}"
      )

    return val["result"]


@dataclass
class System:
    """A class for storing, accessing, managing, and manipulating a molecular
    system's structure, sequence, and topological information. The class is
    organized as a hierarchy of objects:

    System: top-level class containing all information about a molecular system
    -> Chain: a sub-portion of the System; for polymers this is generally a
              chemically connected molecular graph belong to a System (e.g., for
              protein complexes, this would be one of the proteins).
       -> Residue: a generally chemically-connected molecular unit (for polymers,
                   the repeating unit), belonging to a Chain.
          -> Atom: an atom belonging to a Residue with zero, one, or more locations.
             -> AtomLocation: the location of an Atom (3D coordinates and other information).

     Attributes:
         name (str): given name for System
         _chains (list): a list of Chain objects
         _entities (dict): a dictionary of SystemEntity objects, with keys being entity IDs
         _chain_entities (list): `chain_entities[ci]` stores entity IDs (i.e., keys into
             `entities`) corresponding to the entity for chain `ci`
         _extra_models (list): a list of hierarchicList object, representing locations
             for alternative models
         _labels (dict): a dictionary of residue labels. A label is a string value,
             under some category (also a string), associated with a residue. E.g.,
             the category could be "SSE" and the value could be "H" or "S". If entry
             `labels[category][gti]` exists and is equal to `value`, this means that
             residue with global template index `gti` has the label `category:value`.
         _selections (dict): a dictionary of selections. Keys are selection names and
             values are lists of corresponding gti indices.
         _assembly_info (SystemAssemblyInfo): information on symmetric assemblies that can
             be constructed from components of the molecular system. See ``SystemAssemblyInfo``.
    """

    name: str
    _selections: Dict[str, List[int]]

    def __init__(self, name: str = "system"):
        self.name = name
        self._selections = dict()

    def _selex_eval(self, _selex_info, op: str, left, right):
        def _is_numeric(string: str) -> bool:
            try:
                float(string)
                return True
            except ValueError:
                return False

        def _is_int(string: str) -> bool:
            try:
                int(string)
                return True
            except ValueError:
                return False

        def _unpack_operands(operands, dests):
            assert len(operands) == len(dests)
            unpacked = [None] * len(operands)
            succ = True
            for i, (operand, dest) in enumerate(zip(operands, dests)):
                if dest is None:
                    if operand is not None:
                        succ = False
                        break
                elif dest == "result":
                    if not (isinstance(operand, dict) and "result" in operand):
                        succ = False
                        break
                    unpacked[i] = operand["result"]
                elif dest == "string":
                    if not (len(operand) == 1 and isinstance(operand[0], str)):
                        succ = False
                        break
                    unpacked[i] = operand[0]
                elif dest == "strings":
                    if not (
                        isinstance(operand, list)
                        and all([isinstance(val, str) for val in operands])
                    ):
                        succ = False
                        break
                    unpacked[i] = operands
                elif dest == "float":
                    if not (len(operand) == 1 and _is_numeric(operand[0])):
                        succ = False
                        break
                    unpacked[i] = float(operand[0])
                elif dest == "floats":
                    if not (
                        isinstance(operand, list)
                        and all([_is_numeric(val) for val in operands])
                    ):
                        succ = False
                        break
                    unpacked[i] = [float(val) for val in operands]
                elif dest == "range":
                    test = _parse_range(operand)
                    if test is None:
                        succ = False
                        break
                    unpacked[i] = test
                elif dest == "int":
                    if not (len(operand) == 1 and _is_int(operand[0])):
                        succ = False
                        break
                    unpacked[i] = int(operand[0])
                elif dest == "ints":
                    if not (
                        isinstance(operand, list)
                        and all([_is_int(val) for val in operands])
                    ):
                        succ = False
                        break
                    unpacked[i] = [int(val) for val in operands]
                elif dest == "int_range":
                    test = _parse_int_range(operand)
                    if test is None:
                        succ = False
                        break
                    unpacked[i] = test
                elif dest == "int_range_string":
                    test = _parse_int_range(operand, string=True)
                    if test is None:
                        succ = False
                        break
                    unpacked[i] = test
            return unpacked, succ

        def _parse_range(operands: list):
            """Parses range information given a list of operands that were originally separated
            by spaces. Allowed range expressiosn are of the form: `< n`, `> n`, `n:m` with
            optional spaces allowed between operands."""
            if not (
                isinstance(operands, list)
                and all([isinstance(opr, str) for opr in operands])
            ):
                return None
            operand = "".join(operands)
            if operand.startswith(">") or operand.startswith("<"):
                if not _is_numeric(operand[1:]):
                    return None
                num = float(operand[1:])
                if operand.startswith(">"):
                    test = lambda x, cut=num: x > cut
                else:
                    test = lambda x, cut=num: x < cut
            elif ":" in operand:
                parts = operand.split(":")
                if (len(parts) != 2) or not all([_is_numeric(p) for p in parts]):
                    return None
                parts = [float(p) for p in parts]
                test = lambda x, lims=parts: lims[0] < x < lims[1]
            elif _is_numeric(operand):
                target = float(operand)
                test = lambda x, t=target: x == t
            else:
                return None
            return test

        def _parse_int_range(operands: list, string: bool = False):
            """Parses range of integers information given a list of operands that were
            originally separated by spaces. Allowed range expressiosn are of the form:
            `n`, `n-m`, `n+m`, with optional spaces allowed anywhere and combinations
            also allowed (e.g., "n+m+s+r-p+a")."""
            if not (
                isinstance(operands, list)
                and all([isinstance(opr, str) for opr in operands])
            ):
                return None
            operand = "".join(operands)
            operands = operand.split("+")
            ranges = []
            for operand in operands:
                m = re.fullmatch("(.*\d)-(.+)", operand)
                if m:
                    if not all([_is_int(g) for g in m.groups()]):
                        return None
                    r = range(int(m.group(1)), int(m.group(2)) + 1)
                    ranges.append(r)
                else:
                    if not _is_int(operand):
                        return None
                    if string:
                        ranges.append(set([operand]))
                    else:
                        ranges.append(set([int(operand)]))
            if string:
                ranges = [[str(x) for x in r] for r in ranges]
            test = lambda x, ranges=ranges: any([x in r for r in ranges])
            return test

        # evaluate expression and store result in list `result`
        result = set()
        if op in ("and", "or"):
            (Si, Sj), succ = _unpack_operands([left, right], ["result", "result"])
            if not succ:
                return None
            if op == "and":
                result = set(Si).intersection(set(Sj))
            else:
                result = set(Si).union(set(Sj))
        elif op == "not":
            (_, S), succ = _unpack_operands([left, right], [None, "result"])
            if not succ:
                return None
            result = _selex_info["all_indices_set"].difference(S)
        elif op == "all":
            (_, _), succ = _unpack_operands([left, right], [None, None])
            if not succ:
                return None
            result = _selex_info["all_indices_set"]
        elif op == "none":
            (_, _), succ = _unpack_operands([left, right], [None, None])
            if not succ:
                return None
        elif op == "around":
            (S, rad), succ = _unpack_operands([left, right], ["result", "float"])
            if not succ:
                return None

            # Convert to numpy for distance calculation
            atom_indices = np.asarray(
                [
                    ai.aix
                    for ai in _selex_info["all_atoms"]
                    for xi in ai.atom.locations()
                ]
            )
            X_i = np.asarray(
                [
                    [xi.x, xi.y, xi.z]
                    for ai in _selex_info["all_atoms"]
                    for xi in ai.atom.locations()
                ]
            )
            X_j = np.asarray(
                [
                    [xi.x, xi.y, xi.z]
                    for j in S
                    for xi in _selex_info["all_atoms"][j].atom.locations()
                ]
            )
            D = np.sqrt(((X_j[np.newaxis, :, :] - X_i[:, np.newaxis, :]) ** 2).sum(-1))
            ix_match = (D <= rad).sum(1) > 0
            match_hits = atom_indices[ix_match]
            result = set(match_hits.tolist())
        elif op == "saround":
            (S, srad), succ = _unpack_operands([left, right], ["result", "int"])
            if not succ:
                return None
            for j in S:
                aj = _selex_info["all_atoms"][j]
                rj = aj.rix
                for ai in _selex_info["all_atoms"]:
                    if aj.atom.residue.chain != ai.atom.residue.chain:
                        continue
                    ri = ai.rix
                    if abs(ri - rj) <= srad:
                        result.add(ai.aix)
        elif op == "byres":
            (_, S), succ = _unpack_operands([left, right], [None, "result"])
            if not succ:
                return None
            gtis = set()
            for j in S:
                gtis.add(_selex_info["all_atoms"][j].rix)
            for a in _selex_info["all_atoms"]:
                if a.rix in gtis:
                    result.add(a.aix)
        elif op == "bychain":
            (_, S), succ = _unpack_operands([left, right], [None, "result"])
            if not succ:
                return None
            cixs = set()
            for j in S:
                cixs.add(_selex_info["all_atoms"][j].cix)
            for a in _selex_info["all_atoms"]:
                if a.cix in cixs:
                    result.add(a.aix)
        elif op in ("first", "last"):
            (_, S), succ = _unpack_operands([left, right], [None, "result"])
            if not succ:
                return None
            if op == "first":
                mi = min([_selex_info["all_atoms"][i].aix for i in S])
            else:
                mi = max([_selex_info["all_atoms"][i].aix for i in S])
            result.add(mi)
        elif op == "name":
            (_, name), succ = _unpack_operands([left, right], [None, "string"])
            if not succ:
                return None
            for a in _selex_info["all_atoms"]:
                if a.atom.name == name:
                    result.add(a.aix)
        elif op in ("re", "hyd"):
            if op == "re":
                (_, regex), succ = _unpack_operands([left, right], [None, "string"])
            else:
                (_, _), succ = _unpack_operands([left, right], [None, None])
                regex = "[0123456789]?H.*"
            if not succ:
                return None
            ex = re.compile(regex)
            for a in _selex_info["all_atoms"]:
                if a.atom.name is not None and ex.fullmatch(a.atom.name):
                    result.add(a.aix)
        elif op in ("chain", "authchain", "segid"):
            (_, match_id), succ = _unpack_operands([left, right], [None, "string"])
            if not succ:
                return None
            if op == "chain":
                prop = "cid"
            elif op == "authchain":
                prop = "authid"
            elif op == "segid":
                prop = "segid"
            for a in _selex_info["all_atoms"]:
                if getattr(a.atom.residue.chain, prop) == match_id:
                    result.add(a.aix)
        elif op == "resid":
            (_, test), succ = _unpack_operands([left, right], [None, "int_range"])
            if not succ:
                return None
            for a in _selex_info["all_atoms"]:
                if test(a.atom.parent.id[1]):
                    result.add(a.aix)
        elif op in ("resname", "icode"):
            (_, match_id), succ = _unpack_operands([left, right], [None, "string"])
            if not succ:
                return None
            if op == "resname":
                prop = "name"
            elif op == "icode":
                prop = "icode"
            for a in _selex_info["all_atoms"]:
                if getattr(a.atom.residue, prop) == match_id:
                    result.add(a.aix)
        elif op == "authresid":
            (_, test), succ = _unpack_operands(
                [left, right], [None, "int_range_string"]
            )
            if not succ:
                return None
            for a in _selex_info["all_atoms"]:
                if test(a.atom.residue.authid):
                    result.add(a.aix)
        elif op == "gti":
            (_, test), succ = _unpack_operands([left, right], [None, "int_range"])
            if not succ:
                return None
            for a in _selex_info["all_atoms"]:
                if test(a.rix):
                    result.add(a.aix)
        elif op in ("x", "y", "z", "b", "occ"):
            (_, test), succ = _unpack_operands([left, right], [None, "range"])
            if not succ:
                return None
            prop = op
            if op == "b":
                prop = "B"
            for a in _selex_info["all_atoms"]:
                for loc in a.atom.locations():
                    if test(getattr(loc, prop)):
                        result.add(a.aix)
                        break
        elif op == "namesel":
            (_, selname), succ = _unpack_operands([left, right], [None, "string"])
            if not succ:
                return None
            if selname not in self._selections:
                return None
            gtis = set(self._selections[selname])
            for a in _selex_info["all_atoms"]:
                if a.rix in gtis:
                    result.add(a.aix)
        else:
            return None

        return {"result": result}

    def _select(
        self,
        model: Model,
        expression: str,
        unstructured: bool = False,
        left_associativity: bool = True,
    ):
        # Build some helpful data structures to support _selex_eval
        @dataclass(frozen=True)
        class MappableAtom:
            atom: AtomView
            aix: int
            rix: int
            cix: int

            def __hash__(self) -> int:
                return self.aix

        # first collect all real atoms
        all_atoms = [None] * len(list(model.get_atoms()))
        cix, rix, aix = 0, 0, 0
        for chain in model.get_chains():
            for residue in chain.get_residues():
                for atom in residue.get_atoms():
                    all_atoms[aix] = MappableAtom(atom, aix, rix, cix)
                    aix = aix + 1

                # for residues that do not have atoms, add a dummy atom with no location
                # or name; that way, we can still select the residue even though selection
                # algebra fundamentally works on atoms
                if len(list(residue.get_atoms())) == 0:
                    view = DummyAtomView(residue)
                    view.dummy = True
                    # make more room at the end of the list since as this is an "extra" atom
                    all_atoms.append(None)
                    all_atoms[aix] = MappableAtom(view, aix, rix, cix)
                    aix = aix + 1
                rix = rix + 1
            cix = cix + 1

        _selex_info = {"all_atoms": all_atoms}
        _selex_info["all_indices_set"] = set([a.aix for a in all_atoms])

        # fmt: off
        # make an expression tree object
        tree = ExpressionTreeEvaluator(
            ["hyd", "all", "none"],
            ["not", "byres", "bychain", "first", "last",
             "chain", "authchain", "segid", "namesel", "gti", "resix", "resid",
             "authresid", "resname", "re", "x", "y", "z", "b", "icode", "name"],
            ["and", "or", "around", "saround"],
            eval_function=functools.partial(self._selex_eval, _selex_info),
            left_associativity=left_associativity,
            debug=False,
        )
        # fmt: on

        # evaluate the expression
        val = tree.evaluate(expression)

        # if we are not looking to select unstructured residues, remove any dummy
        # atoms. NOTE: making dummy atoms can still impact what structured atoms
        # are selected (e.g., consider `saround` relative to an unstructured residue)
        if not unstructured:
            val = {
                i for i in val if not hasattr(_selex_info["all_atoms"][i].atom, "dummy")
            }

        return val, _selex_info

    def select_residues(
        self,
        model: Model,
        expression: str,
        gti: bool = False,
        allow_unstructured=False,
        left_associativity: bool = True,
    ):
        """Evalates the given selection expression and returns all residues with any
           atoms involved in the result as a list of ResidueView's or list of gti's.

        Args:
            expression (str): selection expression.
            gti (bool): if True (default is False), will return a list of gti
                instead of a list of ResidueView's.
            allow_unstructured (bool): If True (default is False), will allow
                unstructured residues to be selected.
            left_associativity (bool, optional): determines whether operators
                in the expression are left-associative.

        Returns:
            List of ResidueView's or gti's (ints).
        """
        val, selex_info = self._select(
            model,
            expression,
            unstructured=allow_unstructured,
            left_associativity=left_associativity,
        )

        # make a list of ResidueView or gti's
        if gti:
            result = sorted(set([selex_info["all_atoms"][i].rix for i in val]))
        else:
            residues = dict()
            for i in val:
                a = selex_info["all_atoms"][i]
                residues[a.rix] = a.atom.residue
            result = [residues[rix] for rix in sorted(residues.keys())]

        return result


_system = System()

def select_residues(
    model: Model,
    expression: str,
    gti: bool = False,
    allow_unstructured=False,
    left_associativity: bool = True,
):
  return _system.select_residues(
      model,
      expression,
      gti=gti,
      allow_unstructured=allow_unstructured,
      left_associativity=left_associativity
  )

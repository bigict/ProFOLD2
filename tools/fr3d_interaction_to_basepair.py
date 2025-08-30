"""Tools for processing FR3D dataset"""
import os
from collections import defaultdict
import csv

bptypes = (
    "cWW", "tWW", "cWH", "tWH", "cWS", "tWS", "cHH", "tHH", "cHS", "tHS", "cSS", "tSS"
)


def bptype_normalize(bptype):
  assert len(bptype) in (3, 4), bptype
  if len(bptype) == 4:
    if bptype.startswith("n"):
      bptype = bptype[1:]
    elif bptype.endswith("a"):
      bptype = bptype[:-1]
  assert len(bptype) == 3, bptype
  if bptype in ("cHW", "tHW", "cSW", "tSW", "cSH", "tSH"):
    bptype = bptype[0] + bptype[2] + bptype[1]
  return bptype


def unit_id_filter(unit_id):
  if unit_id == "placeholder":
    return False
  return True


def unit_id_split(unit_id):
  return unit_id.split("|")[:5]


def pid_list_read(f):
  x = defaultdict(set)
  for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
    pid, chain = line.split("_", 1)
    x[pid.upper()].add(chain)
  return x


def mapping_idx_read(f):
  x = {}
  for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
    center, v, mol_type = line.split()
    del mol_type
    x[v] = center
  return x


def pid_list_chain_filter(pid_list, entry_id, asym_id):
  if pid_list:
    return entry_id in pid_list and asym_id in pid_list[entry_id]
  return True


def pid_list_unit_filter(pid_list, unit_id1, unit_id2):
  if not all(map(unit_id_filter, (unit_id1, unit_id2))):
    return False

  if pid_list:
    # RNA only.
    unit_id1, unit_id2 = map(unit_id_split, (unit_id1, unit_id2))
    if int(unit_id1[1]) != 1 and int(unit_id2[1]) != 1:
      return False
    if not (unit_id1[0] in pid_list and unit_id2[0] in pid_list):
      return False
    if not (
        unit_id1[2] in pid_list[unit_id1[0]] and unit_id2[2] in pid_list[unit_id2[0]]
    ):
      return False

  return True


def comp_id_list_read(f):
  x = {}
  for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
    pid, mol_type, label_seq_id, label_comp_id, auth_seq_id, _ = line.split()
    # if label_seq_id != auth_seq_id:
    if mol_type in ("mol:rna", "mol:dna"):
      x[(pid, int(auth_seq_id))] = (int(label_seq_id), label_comp_id)
  return x


def comp_id_list_correct(comp_id_list, mapping_idx):
  label_seq_list = defaultdict(dict)
  for (pid, auth_seq_id), (label_seq_id, label_comp_id) in comp_id_list.items():
    label_seq_list[pid][label_seq_id] = label_comp_id

  label_seq_list = {
      pid: sorted(seq_list.items(), key=lambda x: x[0])
      for pid, seq_list in label_seq_list.items()
  }
  label_seq_order = {
      pid: {label_seq_id: idx for idx, (label_seq_id, _) in enumerate(seq_list)}
      for pid, seq_list in label_seq_list.items()
  }

  for (pid, auth_seq_id), (label_seq_id, _) in comp_id_list.items():
    cluster_id = mapping_idx.get(pid, pid)
    if cluster_id != pid:
      assert len(label_seq_list[cluster_id]) == len(label_seq_list[pid]), (
          pid, cluster_id
      )
      comp_id_list[(pid, auth_seq_id)] = label_seq_list[
          cluster_id
      ][label_seq_order[pid][label_seq_id]]
  return comp_id_list


def comp_id_list_lookup(
    comp_id_list, entry_id, asym_id, auth_seq_id, default_comp_id=None
):
  if comp_id_list:
    pid = f"{entry_id.lower()}_{asym_id}"
    return comp_id_list.get((pid, auth_seq_id), (auth_seq_id, default_comp_id))
  return auth_seq_id, default_comp_id


def pseudo_seq_create(
    seq_dict, asym_id1, asym_list, seq_id_start=None, seq_index_gap=128
):
  assert asym_id1 in seq_dict and asym_id1 in asym_list

  def _seq_create(seq_idx, asym_id, aa_idx=1):
    aa_prev = seq_id_start.get(asym_id) if seq_id_start else None

    for i, aa in sorted(seq_dict[asym_id].items(), key=lambda x: x[0]):
      if aa_prev is not None:
        assert i - aa_prev >= 0, (i, aa_prev, asym_id, seq_id_start)
        aa_idx += i - aa_prev
      seq_idx[(asym_id, aa, i)] = aa_idx
      aa_prev = i

    return seq_idx, aa_idx

  seq_idx = defaultdict(int)
  seq_idx, aa_idx = _seq_create(seq_idx, asym_id1, aa_idx=1)

  gap_idx = 1
  for asym_id in seq_dict:
    if asym_id == asym_id1:
      continue
    if not asym_id in asym_list:
      continue
    seq_idx, aa_idx = _seq_create(
        seq_idx, asym_id, aa_idx=aa_idx + gap_idx * seq_index_gap
    )
    gap_idx += 1

  return seq_idx


def bpseq_dict_create(bpseq_dict, bpseq, asym_id1):
  for (
      auth_seq_id1, label_comp_id1, asym_id2, auth_seq_id2, label_comp_id2, basepair
  ) in bpseq:
    bpseq_dict[(asym_id1, label_comp_id1, auth_seq_id1)].add(
        (asym_id2, label_comp_id2, auth_seq_id2, basepair)
    )
    bpseq_dict[(asym_id2, label_comp_id2, auth_seq_id2)].add(
        (asym_id1, label_comp_id1, auth_seq_id1, basepair)
    )
  return bpseq_dict


def bpseq_data_header(entry_id, asym_list):
  return [f"# _entry.id:{entry_id}_{','.join(asym_list)}"]


def bpseq_data_create(
    entry_id, seq_dict, pair_dict, seq_id_start, multimer_threshold=0
):
  def _multimer_filter(bpseq, asym_id1):
    assert bpseq
    p = len([_ for _, _, asym_id2, *_ in bpseq if asym_id2 != asym_id1])
    return p > len(bpseq) * multimer_threshold

  bpseq_dict = defaultdict(set)
  for asym_id, bpseq in pair_dict.items():
    bpseq_dict = bpseq_dict_create(bpseq_dict, bpseq, asym_id)

  for asym_id, bpseq in pair_dict.items():
    asym_list = [asym_id]
    if _multimer_filter(bpseq, asym_id):
      asym_list += list(set([asym for _, _, asym, *_ in bpseq if asym != asym_id]))  # pylint: disable=consider-using-set-comprehension
    assert asym_id in asym_list
    seq_idx = pseudo_seq_create(
        seq_dict, asym_id, asym_list, seq_id_start=seq_id_start.get(entry_id.lower())
    )

    bpseq_data = bpseq_data_header(entry_id, asym_list)  # comment

    for (asym_id1, label_comp_id1, auth_seq_id1), new_idx1 in sorted(
        seq_idx.items(), key=lambda x: x[1]
    ):
      k1 = f"{asym_id1}|{label_comp_id1}|{auth_seq_id1}"
      if (asym_id1, label_comp_id1, auth_seq_id1) in bpseq_dict:
        for asym_id2, label_comp_id2, auth_seq_id2, basepair in bpseq_dict[
            (asym_id1, label_comp_id1, auth_seq_id1)
        ]:
          new_idx2 = seq_idx[(asym_id2, label_comp_id2, auth_seq_id2)]
          k2 = f"{asym_id2}|{label_comp_id2}|{auth_seq_id2}"
          bpseq_data.append(
              f"{new_idx1} {label_comp_id1} {new_idx2} {k1},{k2} {basepair}"
          )
      else:
        bpseq_data.append(f"{new_idx1} {label_comp_id1} 0 {k1}")

    yield entry_id, asym_id, bpseq_data


def base_pairing(rows, pid_list, comp_id_list, multimer_threshold=0):
  seq_id_start = defaultdict(dict)
  for (pid, _), (label_seq_id, _) in comp_id_list.items():
    entry_id, asym_id = pid.split("_")
    if asym_id in seq_id_start[entry_id]:
      seq_id_start[entry_id][asym_id] = min(
          seq_id_start[entry_id][asym_id], label_seq_id
      )
    else:
      seq_id_start[entry_id][asym_id] = label_seq_id

  entry_id, asym_list = None, set()
  seq_dict, pair_dict = defaultdict(dict), defaultdict(set)

  for row in rows:
    unit_id1, unit_id2 = row["unit_id1"], row["unit_id2"]
    basepair = row["FR3D basepair (f_lwbp)"]

    entry_id1, _, asym_id1, label_comp_id1, auth_seq_id1 = unit_id_split(unit_id1)
    entry_id2, _, asym_id2, label_comp_id2, auth_seq_id2 = unit_id_split(unit_id2)
    auth_seq_id1, auth_seq_id2 = map(int, (auth_seq_id1, auth_seq_id2))
    auth_seq_id1, label_comp_id1 = comp_id_list_lookup(
        comp_id_list, entry_id1, asym_id1, auth_seq_id1, default_comp_id=None
    )
    auth_seq_id2, label_comp_id2 = comp_id_list_lookup(
        comp_id_list, entry_id2, asym_id2, auth_seq_id2, default_comp_id=None
    )
    assert entry_id1 == entry_id2, (entry_id1, entry_id2)

    if entry_id != entry_id1:
      if entry_id:
        yield from bpseq_data_create(
            entry_id,
            seq_dict,
            pair_dict,
            seq_id_start,
            multimer_threshold=multimer_threshold
        )
        for asym_id in asym_list:
          if not asym_id in pair_dict:
            yield entry_id, asym_id, bpseq_data_header(entry_id, [asym_id])

      entry_id, asym_list = entry_id1, set()
      seq_dict, pair_dict = defaultdict(dict), defaultdict(set)

    if pid_list_chain_filter(pid_list, entry_id1, asym_id1):
      asym_list.add(asym_id1)
    if pid_list_chain_filter(pid_list, entry_id2, asym_id2):
      asym_list.add(asym_id2)

    if label_comp_id1 is None or label_comp_id2 is None:
      continue

    if pid_list_unit_filter(pid_list, unit_id1, unit_id2):
      seq_dict[asym_id1][auth_seq_id1] = label_comp_id1
      seq_dict[asym_id2][auth_seq_id2] = label_comp_id2

      if basepair:
        pair_dict[asym_id1].add(
            (
                auth_seq_id1, label_comp_id1, asym_id2, auth_seq_id2, label_comp_id2,
                bptype_normalize(basepair)
            )
        )
        print(f"{unit_id1}\t{unit_id2}\t{basepair}\t{bptype_normalize(basepair)}")

  if entry_id:
    yield from bpseq_data_create(
        entry_id,
        seq_dict,
        pair_dict,
        seq_id_start,
        multimer_threshold=multimer_threshold
    )
    for asym_id in asym_list:
      if not asym_id in pair_dict:
        yield entry_id, asym_id, bpseq_data_header(entry_id, [asym_id])


def main(args):  # pylint: disable=redefined-outer-name
  os.makedirs(args.output_dir, exist_ok=True)

  if args.pid_list:
    with open(args.pid_list, "r") as f:
      pid_list = pid_list_read(f)
  else:
    pid_list = None

  if args.comp_id_list:
    with open(args.comp_id_list, "r") as f:
      comp_id_list = comp_id_list_read(f)
    if args.mapping_idx:
      with open(args.mapping_idx, "r") as f:
        mapping_idx = mapping_idx_read(f)
      comp_id_list = comp_id_list_correct(comp_id_list, mapping_idx)
  else:
    comp_id_list = None

  for interaction_file in args.interaction_file:
    with open(interaction_file, "r") as f:
      reader = csv.DictReader(f)

      for entry_id, asym_id, bpseq_data in base_pairing(
          reader, pid_list, comp_id_list, multimer_threshold=args.multimer_threshold
      ):
        if args.standardize_pid:
          entry_id = entry_id.lower()
        with open(
            os.path.join(args.output_dir, f"{entry_id}_{asym_id}.bpseq"), "w"
        ) as f:
          f.write("\n".join(bpseq_data))


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument("interaction_file", type=str, nargs="+", help="interaction file.")
  parser.add_argument("-o", "--output_dir", type=str, default=".", help="output dir.")
  parser.add_argument(
      "-l", "--pid_list", type=str, default=None, help="filter by pid list."
  )
  parser.add_argument(
      "-c",
      "--comp_id_list",
      type=str,
      default=None,
      help="from auth_comp_id to label_comp_id."
  )
  parser.add_argument(
      "-m", "--mapping_idx", type=str, default=None, help="seq mapping index."
  )
  parser.add_argument(
      "-t", "--multimer_threshold", type=float, default=0.0, help="multimer threshold."
  )
  parser.add_argument(
      "--standardize_pid", action="store_true", help="make pid standarded."
  )
  parser.add_argument("-v", "--verbose", action="store_true", help="verbose.")

  args = parser.parse_args()

  main(args)

"""task form module
"""
import io
import re

from profold2.common import residue_constants
from profold2.data.parsers import parse_fasta
from web import db

_JOBID_RE = re.compile('[a-zA-Z0-9_]+', re.I)
_EMAIL_RE = re.compile('^[^@\\s]+@([-a-z0-9]+\\.)+[a-z]{2,}$', re.I)


def bytes_to_string(byte_values):
  t = io.TextIOWrapper(io.BytesIO(byte_values))
  return t.read()


def var_get(var, args, files=None, defval=None, func=lambda x: x.strip()):
  val = defval

  var_file = None
  if isinstance(var, tuple):
    var, var_file = var
  if not val and var in args:
    val = args[var]
  if not val and files and var_file and var_file in files:
    f = files[var_file]
    val = bytes_to_string(f.read())
    f.seek(0)

  if val and func:
    val = func(val)
  return val


def validate(args, files=None):
  erros = []

  # validate job id
  job_id = var_get('job_id', args, files=files)
  if job_id:
    if len(job_id) < 4:
      erros.append('The length of `Job ID` should be at least 4!')
    elif len(job_id) > 20:
      erros.append('The length of `Job ID` should be less than 20!')
    elif not _JOBID_RE.match(job_id):
      erros.append('`Job ID` should only contains alphanumeric characters and \'_\'!')  # pylint: disable=line-too-long
    elif db.job_get(job_id=job_id):
      erros.append('`Job ID` is already used!')

  # validate sequence
  try:
    sequences = var_get(
        ('sequences', 'sequence_file'), args, files=files, func=parse_fasta
    )
    if sequences:
      n = 0
      sequences, descriptions = sequences
      for sequence, description in zip(sequences, descriptions):
        if len(sequence) > 1024:
          erros.append('each `protein sequence` length should be less than 1024!')  # pylint: disable=line-too-long
        if len(sequence) <= 0:
          erros.append('each `protein sequence` length should NOT be EMPTY!')
        if len(description) > 1024:
          erros.append('each `protein id` length should be less than 1024!')
        for j, aa in enumerate(sequence):
          if not aa in residue_constants.restype_order:
            i, k = min(0, j - 20), max(len(sequence), j + 20)
            erros.append((f'In the following sequence, the blue character is invalid.<br>'  # pylint: disable=line-too-long
                          f'{description}<br>'
                          f'...{sequence[i:j]}<font color="blue">{aa}</font>{sequence[j+1:k]} ...'))  # pylint: disable=line-too-long
        n += len(sequence) + len(description) + 2
      if n >= (1 << 16):
        erros.append(f'`input sequences` length should be less than {1<<6}k')
    if not sequences:
      erros.append('`sequences` requires and should be FASTA!')
  except Exception as e:  # pylint: disable=broad-except
    erros.append(f'`sequences` requires and should be FASTA! {e}')

  # validate email
  email = var_get('email', args, files=files)
  if email and not _EMAIL_RE.match(email):
    erros.append('The `email` address must be a valid email!')

  return erros

"""
Module for ProFOLD default handler
"""

import os
import collections
import functools
import importlib
import io
import base64
import json
import logging

import torch
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import list_classes_from_module

from profold2.data.dataset import ProteinSequenceDataset
from profold2.model import ReturnValues
from profold2.model.features import FeatureBuilder
from profold2.data.parsers import parse_fasta
from profold2.data.utils import pdb_save

logger = logging.getLogger(__file__)

class FastaHandler(BaseHandler):  # pylint: disable=missing-class-docstring
  def initialize(self, context):
    """Initialize function loads the model.pt file and initialized the model
     object.
       First try to load torchscript else load eager mode state_dict based
     model.
    Args:
        context (context): It is a JSON Object containing information
        pertaining to the model artifacts parameters.
    Raises:
        RuntimeError: Raises the Runtime error when the model.py is missing
    """
    #super().initialize(context)
    properties = context.system_properties
    self.map_location = 'cuda' if torch.cuda.is_available(
    ) and properties.get('gpu_id') is not None else 'cpu'
    self.device = torch.device(
        self.map_location + ':' + str(properties.get('gpu_id'))
        if torch.cuda.is_available() and properties.get('gpu_id') is not None
        else self.map_location
    )
    self.manifest = context.manifest

    model_dir = properties.get('model_dir')
    model_pt_path = None
    if 'serializedFile' in self.manifest['model']:
      serialized_file = self.manifest['model']['serializedFile']
      model_pt_path = os.path.join(model_dir, serialized_file)

    # model def file
    model_file = self.manifest['model'].get('modelFile', '')

    self.feat_builder = None
    if model_file:
      def _feats_gen(feats, device):
        for fn, opts in feats:
          if 'device' in opts:
            opts['device'] = device
          yield fn, opts

      logger.debug('Loading eager model')
      model_args = None
      if os.path.exists(os.path.join(model_dir, 'model_args.json')):
        with open(os.path.join(model_dir, 'model_args.json'), 'r') as f:
          model_args = json.load(f)
      logger.info('model_args: %s', model_args)
      self.model = self._load_pickled_model(
          model_dir, model_file, model_args, model_pt_path)
      self.model.to(self.device)
      if model_args:
        self.feat_builder = FeatureBuilder(
            list(_feats_gen(model_args.get('feats', []), self.device)),
            is_training=False)
    else:
      logger.debug('Loading torchscript model')
      if not os.path.isfile(model_pt_path):
        raise RuntimeError('Missing the model.pt file')

      self.model = self._load_torchscript_model(model_pt_path)

    self.model.eval()
    # if ipex_enabled:
    #   self.model = self.model.to(memory_format=torch.channels_last)
    #   self.model = ipex.optimize(self.model)

    logger.debug('Model file %s loaded successfully', model_pt_path)

    # Load class mapping for classifiers
    # mapping_file_path = os.path.join(model_dir, 'index_to_name.json')
    # self.mapping = load_label_mapping(mapping_file_path)

    self.initialized = True
    logger.info('initialized.')

  def preprocess(self, data):
    sequences, descriptions = [], []

    for row in data:
      fasta = row.get('data') or row.get('body')
      # If the image is sent as bytesarray
      if isinstance(fasta, (bytearray, bytes)):
        f = io.TextIOWrapper(io.BytesIO(fasta))
        fasta = f.read()

      s, d = parse_fasta(fasta)
      sequences += s
      descriptions += d

    dataset = ProteinSequenceDataset(sequences, descriptions)
    loader = torch.utils.data.DataLoader(dataset,
        collate_fn=functools.partial(dataset.collate_fn,
            feat_builder=self.feat_builder),
        batch_size=1,
        shuffle=False)
    return [batch for batch in iter(loader)]  # pylint: disable=unnecessary-comprehension

  def inference(self, data, *args, **kwargs):
    results = []

    with torch.no_grad():
      for i, batch in enumerate(iter(data)):
        logger.info('inference: i=%d, batch=%s', i, batch)
        r =  self.model(batch)
        logger.info(r)
        results.append((i, batch, ReturnValues(**r)))

    return results

  def postprocess(self, data):
    def to_base64(headers):
      with io.BytesIO() as f:
        torch.save(headers, f)
        t = io.TextIOWrapper(io.BytesIO(
            base64.b64encode(f.getvalue())))
        return t.read()
    def to_dict(result):
        i, b, r = result
        return pdb_save(i, b, r.headers, return_pdb=True), to_base64(r.headers)

    results = collections.defaultdict(list)
    for r in data:
      pdb, headers = to_dict(r)
      results['pdb'] += pdb
      results['headers'] += [headers]
      
    return [results]

  def _load_pickled_model(self,
      model_dir, model_file, model_args, model_pt_path):
    """
    Loads the pickle file from the given model path.
    Args:
        model_dir (str): Points to the location of the model artefacts.
        model_file (.py): the file which contains the model class.
        model_pt_path (str): points to the location of the model pickle file.
    Raises:
        RuntimeError: It raises this error when the model.py file is missing.
        ValueError: Raises value error when there is more than one class in the
                    label, since the mapping supports only one label per class.
    Returns:
        serialized model file: Returns the pickled pytorch model file
    """
    model_def_path = os.path.join(model_dir, model_file)
    if not os.path.isfile(model_def_path):
      raise RuntimeError('Missing the model.py file')

    module = importlib.import_module(model_file.split('.')[0])
    model_class_definitions = list_classes_from_module(module)
    if len(model_class_definitions) != 1:
      raise ValueError(
          'Expected only one class as model definition. {}'.format(
              model_class_definitions))

    model_class = model_class_definitions[0]
    model = model_class(model_args)
    if model_pt_path:
      state_dict = torch.load(model_pt_path, map_location=self.device)
      model.load_state_dict(state_dict)
    return model

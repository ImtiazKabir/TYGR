import argparse
import sys

from . import datagen
from . import datamerge
from . import datasplit
from . import dataslice
from . import datastats
from . import dataviz
from . import finetune
from . import predict
from . import train
from . import test
from . import hypergraph_convert
from . import true_hypergraph_convert
from . import view_predictions
from . import compare_types

from . import patch_arch

modules = {
  "datagen": datagen,
  "datamerge": datamerge,
  "datasplit": datasplit,
  "dataslice": dataslice,
  "datastats": datastats,
  "dataviz": dataviz,
  "predict": predict,
  "train": train,
  "test": test,
  "finetune": finetune,

  # Other stuffs
  "patch-arch": patch_arch,
  "hypergraph_convert": hypergraph_convert,
  "true_hypergraph_convert": true_hypergraph_convert,
  "view_predictions": view_predictions,
  "compare_types": compare_types,
}

def parser():
  parser = argparse.ArgumentParser(description="bityr")
  setup_parser(parser)
  return parser

def setup_parser(parser):
  subparsers = parser.add_subparsers(dest="cmd", required=True)
  for (key, module) in modules.items():
    subparser = subparsers.add_parser(key)
    module.setup_parser(subparser)

def main(args):
  sys.setrecursionlimit(5000)
  if args.cmd in modules:
    modules[args.cmd].main(args)
  else:
    raise Exception(f"Unknown command {args.cmd}")

if __name__ == '__main__':
  args = parser().parse_args()
  main(args)

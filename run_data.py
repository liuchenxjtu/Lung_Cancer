#!/usr/bin/env python
#
#   Copyright 2017 Anil Thomas
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""
Train and validate a model

Usage:
    ./run.py -w </path/to/data> -e 8 -r 0
"""
import os
import numpy as np
from neon.util.argparser import NeonArgparser
from data import ChunkLoader
import settings
import video


# Parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('-tm', '--test_mode', action='store_true',
                    help='make predictions on test data')
args = parser.parse_args()

# Setup data provider
repo_dir = args.data_dir
common = dict(datum_dtype=np.uint8, repo_dir=repo_dir, test_mode=args.test_mode)
test = ChunkLoader(set_name='test', augment=not args.test_mode, **common)
print "# batches", test.nbatches

for uid, data, targets, starts in test:
    print uid
    print data.get()
    print targets.get()
    print starts.get()
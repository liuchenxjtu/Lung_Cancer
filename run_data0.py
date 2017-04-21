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
from data0 import ChunkLoader
import settings
import video

# Parse the command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--s", help="source data folder")

parser.add_argument("--d", help="destination data folder")
parser.add_argument("--tm", help="test mode")
parser.add_argument("--set_name", help="set name")
args = parser.parse_args()

# Setup data provider
repo_dir = args.s
save_folder = args.d
common = dict(datum_dtype=np.uint8, repo_dir=repo_dir, test_mode=args.tm)
set_name= args.set_name

test = ChunkLoader(set_name=set_name, **common)
labels_file = open(save_folder+"labels.txt",'w')
for i in range(test.data_size):
    c,s,t = test.next_video(i)
    if i%100==0:
        print "procedding ",i
#     print np.array(c).shape,np.array(s).shape,np.array(t).shape
    np.save(save_folder+"locaition_"+test.current_uid,np.array(s))
#     np.save(save_folder+"label_"+test.current_uid,np.array(t))
    np.save(save_folder+"chunk_"+test.current_uid,np.array(c))
    for i,l in enumerate(t):
        print >>labels_file,test.current_uid,i,l
labels_file.close()
# data = ChunkLoader(set_name=set_name, augment=not args.test_mode, **common)
# repo_dir = '/Users/chen.liu/nfs03/share_data/Intelligence/Scoupon/items/luna_new_vids/'
# Setup data provider
# repo_dir = args.data_dir

# common = dict(datum_dtype=np.uint8, repo_dir=repo_dir, test_mode=args.test_mode)
# test = ChunkLoader(set_name='test', augment=not args.test_mode, **common)
# print "# batches", test.nbatches
#
# for uid, data, targets, starts in test:
#     print uid
#     print data.get()
#     print targets.get()
#     print starts.get()

import logging
import numpy as np
import pandas as pd
import os
import video
import settings


class ChunkLoader():
    def __init__(self, set_name, repo_dir, datum_dtype=np.uint8,
                 nclasses=2, augment=False, test_mode=False):
        # assert test_mode is False, 'Test mode not implemented yet'
        np.random.seed(0)
        self.set_name = set_name
        # self.bsz = self.be.bsz
        self.augment = augment
        self.repo_dir = repo_dir
        self.is_training = (set_name == 'train')
        self.chunk_size = settings.chunk_size
        self.chunk_shape = (self.chunk_size, self.chunk_size, self.chunk_size)
        self.chunk_volume = np.prod(self.chunk_shape)
        self.metadata = pd.read_csv(os.path.join(self.repo_dir, set_name + '-metadata.csv'))
        self.data_size = self.metadata.shape[0]
        self.pos_users = self.metadata[self.metadata['flag']==1]['uid']
        self.neg_users = self.metadata[self.metadata['flag']==0]['uid']
        self.nvids = self.metadata.shape[0]
        self.chunks_filled = 0
        self.video_idx = 0
        if not test_mode:
            self.labels = pd.read_csv(os.path.join(self.repo_dir, 'labels.csv'))
            self.nega_labels = pd.read_csv(os.path.join(self.repo_dir, 'candidates.csv'))
        else:
            self.labels = None
        self.test_mode = test_mode
        self.chunks,self.starts,self.targets = [],[],[]
        ##positive points in lables.csv
        self.pos_labels = self.labels[self.labels['uid'].isin(self.pos_users)].shape[0]
        self.pos_neg_ratio = 6.0
        self.chunk_from_neg_users = int(self.pos_labels*self.pos_neg_ratio/len(self.neg_users))
        self.current_uid = self.current_flag = self.current_meta = None

    def reset(self):
        self.chunks,self.starts,self.targets = [],[],[]
    def next_video(self,video_idx):
        self.reset()
        self.current_meta = self.metadata.iloc[video_idx]
        uid = self.current_meta['uid']
        self.current_uid = self.current_meta['uid']
        self.current_flag = int(self.current_meta['flag'])
        data_filename = os.path.join(self.repo_dir, uid + '.' + settings.file_ext)
        vid_shape = (int(self.current_meta['z_len']),
                     int(self.current_meta['y_len']),
                     int(self.current_meta['x_len']))
        vid_data = video.read_blp(data_filename, vid_shape)
        self.video_idx += 1
        self.extract_chunks(vid_data)


        return self.chunks,self.starts,self.targets


    def slice_chunk(self, start, data):
        return data[start[0]:start[0] + self.chunk_size,
               start[1]:start[1] + self.chunk_size,
               start[2]:start[2] + self.chunk_size]#.ravel()

    def extract_one(self, data, data_shape, uid_data,idx):
        # assert uid_data.shape[0] != 0
        if not self.test_mode:
                center = np.array((uid_data['z'].iloc[idx],
                                   uid_data['y'].iloc[idx],
                                   uid_data['x'].iloc[idx]), dtype=np.int32)
                # radius
                rad = 0.5 * uid_data['diam'].iloc[idx]
                if rad == 0:
                    # Assign an arbitrary radius to candidate nodules
                    rad = 20 / settings.resolution

                #comment by lc: low may <0
                low = np.int32(center + rad - self.chunk_size)
                high = np.int32(center - rad)
                for j in range(3):
                    low[j] = max(0, low[j])
                    high[j] = max(low[j] + 1, high[j])
                    high[j] = min(data_shape[j] - self.chunk_size, high[j])
                    low[j] = min(low[j], high[j] - 1)
                start = [np.random.randint(low=low[i], high=high[i]) for i in range(3)]
        else:
            start = self.generate_chunk_start(chunk_idx, data_shape)


        chunk = self.slice_chunk(start, data)

        return chunk,start

    def generate_chunk_start(self, chunk_idx, data_shape):
        chunk_spacing = np.int32((np.array(data_shape) - self.chunk_size) / settings.chunks_per_dim)
        z_chunk_idx = chunk_idx / settings.chunks_per_dim ** 2
        y_chunk_idx = (chunk_idx - z_chunk_idx * settings.chunks_per_dim ** 2) / settings.chunks_per_dim
        x_chunk_idx = chunk_idx - z_chunk_idx * settings.chunks_per_dim ** 2 \
                      - y_chunk_idx * settings.chunks_per_dim

        start = [z_chunk_idx * chunk_spacing[0],
                 y_chunk_idx * chunk_spacing[1],
                 x_chunk_idx * chunk_spacing[2]]
        return start

    def extract_chunks(self, data):
        data_shape = np.array(data.shape, dtype=np.int32)
        if self.current_flag:
            uid_data = self.labels[self.labels['uid'] == self.current_uid]
            for idx in range(uid_data.shape[0]):
                chunk,start = self.extract_one(data, data_shape, uid_data, idx)
                if chunk is None:
                    continue
                self.chunks.append(chunk)
                self.starts.append(start)
                self.targets.append(1)
        else:
            uid_data = self.nega_labels[self.nega_labels['uid'] == self.current_uid]
            for i in range(min(self.chunk_from_neg_users,uid_data.shape[0])):
                idx = np.random.randint(uid_data.shape[0])
                chunk,start = self.extract_one(data, data_shape, uid_data, idx)
                if chunk is None:
                    continue
                self.chunks.append(chunk)
                self.starts.append(start)
                self.targets.append(0)

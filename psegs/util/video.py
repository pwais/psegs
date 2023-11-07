# Copyright 2023 Maintainers of PSegs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import attr
from pathlib import Path

@attr.s(slots=True, eq=True, weakref_slot=False)
class VideoMeta(object):
  video_uri = attr.ib(default='')

  start_time_nanostamp = attr.ib(default=0)
  frames_per_second = attr.ib(default=0.0)
  n_frames = attr.ib(default=0)

  height = attr.ib(default=0)
  width = attr.ib(default=0)
  
  is_10bit_hdr = attr.ib(default=False)

  lstat_nanostamp = attr.ib(default=0)
  lstat_attr = attr.ib(default='st_mtime')

  @classmethod
  def create_for_video(cls, video_uri, lstat_attr='st_mtime'):
    import imageio
    r = imageio.get_reader(video_uri)
    n_frames = r.get_meta_data()['nframes']
    if n_frames == float('inf'):
      # For some python / imageio versions, you have to use this API:
      n_frames = r.count_frames()
      if n_frames == float('inf'):
        # TODO: use ffmpeg directly?
        raise ValueError(
          "Don't currently support infinite streams: %s %s" % (
            r.get_meta_data(), video_uri))
    
    fps = r.get_meta_data()['fps']
    h, w = r.get_data(0).shape[:2]

    lstat_res = Path(video_uri).lstat()
    start_time_sec = getattr(lstat_res, lstat_attr)
    lstat_nanostamp = int(1e9 * start_time_sec)
    start_time_nanostamp = lstat_nanostamp

    is_10bit_hdr = False

    vid_dict = maybe_get_ffmpeg_meta(video_uri)
    if vid_dict:
      start_time_nanostamp = (
        ffmpeg_meta_maybe_get_start_time(vid_dict) or start_time_nanostamp)
      is_10bit_hdr = (
        ffmpeg_meta_maybe_get_is_10bit_hdr(vid_dict) or is_10bit_hdr)

      # TODO try to read:
      # * displaymatrix
      # * com.apple.quicktime.location.ISO6709
      # * Core Media Metadata


    return cls(
            video_uri=video_uri,
            start_time_nanostamp=start_time_nanostamp,
            frames_per_second=fps,
            n_frames=n_frames,
            height=h,
            width=w,
            is_10bit_hdr=is_10bit_hdr,
            lstat_nanostamp=lstat_nanostamp,
            lstat_attr=lstat_attr)


def maybe_get_ffmpeg_meta(video_uri):
  import ffmpeg

  vid_dict = None
  try:
    vid_dict = ffmpeg.probe(video_uri)
  except Exception as e:
    # TODO maybe just log as warning?
    pass

  return vid_dict

def ffmpeg_meta_maybe_get_start_time(vid_dict):
  import dateparser

  date_raw_str = (
    vid_dict['format']['tags'].get('com.apple.quicktime.creationdate') or
    vid_dict['format']['tags'].get('creation_time')
    '')
  
  try:
    return dateparser.parse(date_raw_str)
  except Exception as e:
    return None
  
def ffmpeg_meta_maybe_get_is_10bit_hdr(vid_dict):
  for s in vid_dict.get('streams', []):
    pix_fmt = s.get('pix_fmt', '')
    if pix_fmt.endswith('p10le'):
      return True
  
  return False

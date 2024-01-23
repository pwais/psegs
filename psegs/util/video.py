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
  
  # Note: A video can have dropped frames and/or invalid metadata, so these
  # could disagree with the actual extracted frames (especially `n_frames`)
  n_frames = attr.ib(default=0)
  frames_per_second = attr.ib(default=0.0)
  end_time_nanostamp = attr.ib(default=0)
  height = attr.ib(default=0)
  width = attr.ib(default=0)
  
  is_10bit_hdr = attr.ib(default=False)

  lstat_nanostamp = attr.ib(default=0)
  lstat_attr = attr.ib(default='st_mtime')

  @classmethod
  def create_for_video(cls, video_uri, lstat_attr='st_mtime', prefer_ffmpeg=True):
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
    
    duration_sec = r.get_meta_data().get('duration', 0.0)
    end_time_nanostamp = start_time_nanostamp + duration_sec * 1e9

    is_10bit_hdr = False

    vid_dict = maybe_get_ffmpeg_meta(video_uri)
    if vid_dict:
      if prefer_ffmpeg:
        # imageio and ffmpeg can sometimes disagree on e.g. frame count :(
        for stream in vid_dict.get('streams', []):
          if stream.get('codec_type') == 'video':
            n_frames = int(stream.get('nb_frames', n_frames))
            
            def _to_float(fraction_str):
              try:
                n, d = fraction_str.split('/')
                return float(n) / float(d)
              except ValueError:
                try:
                  return float(fraction_str)
                except Exception:
                  return None
            
            ffmpeg_fps = _to_float(stream.get('avg_frame_rate'))
            fps = ffmpeg_fps or fps

            duration_sec_str = stream.get('duration', '')
            if duration_sec_str:
              duration_sec = None
              try:
                duration_sec = float(duration_sec_str)
              except:
                pass
              if duration_sec is not None:
                end_time_nanostamp = start_time_nanostamp + duration_sec * 1e9

            break

        ffmpeg_start_time_nanostamp = ffmpeg_meta_maybe_get_start_time(vid_dict)
        if ffmpeg_start_time_nanostamp is not None and prefer_ffmpeg:
          start_time_nanostamp = ffmpeg_start_time_nanostamp
      
      is_10bit_hdr = (
        ffmpeg_meta_maybe_get_is_10bit_hdr(vid_dict) or is_10bit_hdr)
      

      # TODO try to read:
      # * com.apple.quicktime.location.ISO6709
      # * Core Media Metadata

    return cls(
            video_uri=video_uri,
            start_time_nanostamp=start_time_nanostamp,
            frames_per_second=fps,
            n_frames=n_frames,
            end_time_nanostamp=end_time_nanostamp,
            height=h,
            width=w,
            is_10bit_hdr=is_10bit_hdr,
            lstat_nanostamp=lstat_nanostamp,
            lstat_attr=lstat_attr)


def maybe_get_ffmpeg_meta(video_uri):
  import ffmpeg
  assert hasattr(ffmpeg, 'probe'), \
    f"probe() function missing, do you have a bad install of `ffmpeg-python`? {ffmpeg.__file__}"

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
    vid_dict['format']['tags'].get('creation_time') or
    '')
  
  if not date_raw_str:
    return None

  try:
    return dateparser.parse(date_raw_str).timestamp()
  except Exception as e:
    return None
  
def ffmpeg_meta_maybe_get_is_10bit_hdr(vid_dict):
  for s in vid_dict.get('streams', []):
    pix_fmt = s.get('pix_fmt', '')
    if pix_fmt.endswith('p10le'):
      return True
  
  return False


@attr.s(slots=True, eq=True, weakref_slot=False)
class VideoExplodeParams(object):

  max_hw = attr.ib(default=-1)

  image_file_extension = attr.ib(default='png')

  jpeg_quality_percent = attr.ib(default=100)

  # Extract only the first `n_frames`
  n_frames = attr.ib(default=-1)


def ffmpeg_explode(params, video_uri, dest_root):
  import math
  from oarphpy import util as oputil
  
  try:
    oputil.run_cmd("ffmpeg -h")
  except Exception as e:
    raise ValueError(f"This functionality requires system ffmpeg, got {e}")

  video_path = Path(video_uri).resolve()

  rescale_arg = ''
  if params.max_hw >= 0:
    rescale_arg = (
      f"-vf 'scale=if(gte(iw\,ih)\,min({params.max_hw}\,iw)\,-2):if(lt(iw\,ih)\,min({params.max_hw}\,ih)\,-2)' "
    )
  qscale_arg = ''
  if params.image_file_extension == 'jpg':
    # ffmpeg jpeg quality is from 2 to 31 with 2 highest
    jpeg_quality = 2 + (31 - 2) * (1. - (params.jpeg_quality_percent / 100.))
    qscale_arg = f" -qscale {jpeg_quality} "

  vframes_arg = ''
  zfill = 6
  if params.n_frames >= 0:
    zfill = int(math.log10(params.n_frames)) + 1
    vframes_arg = f" -vframes {params.n_frames} "

  FFMPEG_CMD = f"""
    cd "{dest_root}" && \
    ffmpeg \
      -y -v quiet -stats \
      -noautorotate \
      -i {video_path} \
      {vframes_arg} \
      {rescale_arg} \
      -vsync 0 \
      {qscale_arg} \
        ffmpeg_explode_frame_%0{zfill}d.{params.image_file_extension}
  """
  oputil.run_cmd(FFMPEG_CMD)

  paths = sorted(
            oputil.all_files_recursive(
              dest_root, 
              pattern='ffmpeg_explode_frame_*'))
  return paths

"""Meaningful regression checks for the isolated notebook experiment harness."""
import csv
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np

from tools.rnd_full_pipeline import (Profiler, compare_event_rows,
                                    deepsort_assignments, profiled_pipeline_module)


def event(t, direction='South'):
    return dict(event_timestamp_s=str(t), travel_direction=direction)


def test_event_agreement_uses_assignment_gating_and_not_track_ids():
    pairs, missing, extra = compare_event_rows(
        [event(1),event(3),event(5)], [event(3.1),event(1.2),event(5,'North')])
    assert sorted(pairs) == [(0,1),(1,0)]
    assert missing == extra == [2]
    assert compare_event_rows([], [event(1)]) == ([],[],[0])


def test_deepsort_rejects_stale_and_tentative_predictions():
    class Track:
        def __init__(self, ident, index, age=0, confirmed=True):
            self.track_id,self.index,self.time_since_update=ident,index,age
            self.confirmed=confirmed
        def is_confirmed(self):return self.confirmed
        def get_det_supplementary(self):return {'detection_index':self.index}
        def to_ltrb(self):return [1,2,20,30]
    class Tracker:
        def update_tracks(self, raw, frame, others):
            assert others == [{'detection_index':0},{'detection_index':1}]
            return [Track('4',0),Track('8',1,age=1),Track('9',1,confirmed=False)]
    class Boxes(SimpleNamespace):
        def __len__(self):return len(self.conf)
    boxes=Boxes(xyxy=np.array([[1,2,20,30],[50,2,80,30]]),conf=np.array([.9,.8]),cls=np.array([0,0]))
    result=deepsort_assignments(Tracker(),boxes,np.zeros((100,100,3),np.uint8))
    assert list(result)==[0]
    assert result[0][4]==4
    assert result[0][-1]==0


def test_timing_hooks_preserve_pipeline_events(tmp_path, config_path):
    import cv2
    from hauling.engine.pipeline import RunOptions,run_pipeline
    from hauling.vision.detection import Detection
    from hauling.vision.plate import PlateCandidate
    class Adapter:
        device='synthetic'
        model_classes={'vehicle':{0:'Truck'},'plate':{0:'Character'}}
        def __init__(self,*args):self.i=-1
        def detect(self,frame):
            self.i+=1
            y=[5,15,35,45,55,65,75,85,85][self.i]
            return [Detection((40,y-4,60,y+4),.9,1,class_id=0,class_name='Truck')],1
        def detect_plate(self,frame,detection,timestamp_s):
            return PlateCandidate(.9,(45,51,55,54),timestamp_s)
    video=tmp_path/'input.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(100,100))
    assert writer.isOpened()
    for _ in range(9):writer.write(np.zeros((100,100,3),np.uint8))
    writer.release()
    vehicle=tmp_path/'v.pt';vehicle.touch()
    hull=tmp_path/'h.pt';hull.touch()
    options=RunOptions(video,vehicle,hull,config_path,tmp_path/'normal',
                       save_video=True,report=True,ocr_lookahead_seconds=.3)
    original=run_pipeline(options,Adapter)
    prof=Profiler(cuda=False)
    wrapped=profiled_pipeline_module().run_pipeline(replace(options,output_dir=tmp_path/'timed'),Adapter,profiler=prof)
    assert (tmp_path/'normal/events.csv').read_bytes()==(tmp_path/'timed/events.csv').read_bytes()
    for key in ['events_csv','speed_valid','overspeed','plate_detected','ocr_windows_truncated','video_saved','report_saved']:
        assert original[key]==wrapped[key]
    assert len(prof.frames)==9
    assert all(r['pipeline_frame_ms']>=r['decode_ms']>=0 for r in prof.frames)

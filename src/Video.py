import cv2
from utils import map_range


class Video(object):
    def __init__(self, path):
        self.path = path
        self.frames = []
        self.name = str.split(self.path, '.')[0].split('/')[-1]
        self.fps = -1

    def load_clip(self, frames_to_load=-1):
        """ Load a video clip in self.path into self.frames. """
        cap = cv2.VideoCapture(self.path)
        frame_ct = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if frames_to_load == -1:
            frames_to_load = frame_ct
        for i in range(frames_to_load):
            ret, frame = cap.read()
            self.frames.append(frame)
        print frames_to_load, "frames loaded. Frame size: {0[0]}x{0[1]}.".format(frame.shape)
        print "Video length:", len(self.frames) / self.fps, "seconds. FPS:", self.fps, "."

    def play_clip(self, fps=120, frames=None):
        """
        Play a video clip. Press Q to quit.
        :param fps: Frames per second. Default is 120.
        :param frames: A tuple of start and end frames.
        """
        if frames is None:
            frames = (0, len(self.frames))
        for frame in range(frames[0], frames[1]):
            cv2.imshow('frame', self.frames[frame])
            cv2.moveWindow("frame", 1200, 0)
            if cv2.waitKey(1000 / fps) & 0xFF == ord('q'):
                break

    def extract_frames_from_timestamps(self, starttime, endtime, fixeddur):
        """ Return start and end frames, given start and end timestamps."""
        mins, secs = str.split(starttime, ':')
        startsecs = int(mins) * 60 + int(secs)
        mins, secs = str.split(endtime, ':')
        endsecs = int(mins) * 60 + int(secs)
        framect = len(self.frames)
        framestart = int(map_range(startsecs, 0, framect / self.fps, 0, framect))
        if fixeddur == -1:
            frameend = int(map_range(endsecs, 0, framect / self.fps, 0, framect))
        else:
            frameend = int(map_range(startsecs + fixeddur, 0, framect / self.fps, 0, framect))
        if self.frames < frameend:
            print "WARNING: Video" + self.name + "is too short."
        return framestart, min(self.frames, frameend)

    def play_clip_timestamps(self, (starttime, endtime), fps=120):
        """
        Play a video clip. Alternative to play_clip. Press Q to quit.
        :param starttime: A string denoting start time MM:SS format.
        :param endtime: A string denoting end time MM:SS format.
        :param fps: Playback frames per second. Default is 120.
        """
        framestart, frameend = self.extract_frames_from_timestamps(starttime, endtime, fixeddur)
        self.play_clip(fps=fps, frames=(framestart, frameend))

    def save_clip_timestamps(self, (starttime, endtime), path, fixeddur=-1):
        """
        Save a sequence of frames as a video clip.
        :param starttime: A string denoting start time MM:SS format.
        :param endtime: A string denoting end time MM:SS format.
        :param path: Destination path for the saved video. (Ex: ../vid/video.MP4)
        :param fixeddur: If fixeddur is not -1, the video is saved starting from starttime, for fixeddur seconds.
        """
        framestart, frameend = self.extract_frames_from_timestamps(starttime, endtime, fixeddur)
        height, width, layers = self.frames[0].shape
        video = cv2.VideoWriter(path, 0x21, 30, (width, height))  # use 0x21 or cv2.VideoWriter_fourcc(*'X264') as CODEC
        for frame in range(framestart, frameend):
            video.write(self.frames[frame])
        cv2.destroyAllWindows()
        video.release()
        print "Video saved as:", path + "."

    def get_video_params_from_name(self):
        """
        Return hive no, tag color, tag number, video ID from video name.
        Video name must be in the format: <HIVE>_<COLOR>_<TAGNUMBER>_<VIDEOID>.
        """
        lst = self.name.split('_')
        hive = lst[1]
        tagc = "yellow" if lst[2][-1] == 'Y' else "white"
        tagid = lst[2][:-1]
        vidid = lst[3]
        return hive, tagc, tagid, vidid

    def tag_clip(self, spreadsheet):
        pass

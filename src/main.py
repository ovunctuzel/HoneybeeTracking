from Video import Video
from Spreadsheet import Spreadsheet
import os




def extract_clip(video, sheet, target_behaviors, dur=-1, save=True, targetdir=""):
    """
    Extract short clips for target behaviors in a video.
    :param video: Video object to extract from.
    :param sheet: Spreadsheet to accompany the video.
    :param dur: Clip duration.
    :param target_behaviors: list of behviors to extract.
    :param save: Save the video clip to harddrive.
    :param targetdir: The target directory for saving the clip.
    """
    events = sheet.get_video_events(video.get_video_params_from_name())
    for i, event in enumerate(events):
        print event['Behavior']
        if event['Behavior'] in target_behaviors:
            outpath = targetdir + event['Behavior'] + str(i) + ".MP4"
            if save:
                video.save_clip_timestamps((event['StartTime'], event['EndTime']), path=outpath, fixeddur=dur)
            else:
                video.play_clip_timestamps((event['StartTime'], event['EndTime']), path=outpath, fixeddur=dur)

def extract_all_clips(folder, sheet, target_behaviors, dur=-1, save=True, targetdir=""):
    """ Extract clips for target behaviors from all videos in a folder. """
    videonames = os.listdir(folder)
    for videoname in videonames:
        if videoname[-3:] == "MP4":
            video = Video(videoname)
            video.load_clip(frames_to_load=-1)
            extract_clip(video, sheet, target_behaviors, dur, save, targetdir)

if __name__ == "__main__":
    videodir = "/media/ovunc/Storage/Honeybee/Videos_Compressed/"
    sheetdir = "/media/ovunc/Storage/Honeybee/Honeybee.csv"
    outdir = "/media/ovunc/Storage/Honeybee/Output_Videos/"
    sheet = Spreadsheet(sheetdir)
    extract_all_clips(videodir, sheet, [], dur=3, save=True, targetdir=outdir)


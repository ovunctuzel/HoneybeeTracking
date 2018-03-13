import csv


class Spreadsheet(object):
    def __init__(self, path):
        self.path = path
        self.items = []

    def load_items(self):
        """
        Load a list of dictionaries into member variable items.
        Each dictionary corresponds to a row in the csv file.
        All items in the dictionary are strings.
        """
        with open(self.path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for entry in reader:
                dct = {}
                names = ['Hive',
                         'TagColor',
                         'TagNumber',
                         'VideoID',
                         'Date',
                         'AMPM',
                         'Viewer',
                         'Behavior',
                         'Initiator',
                         'StartTime',
                         'EndTime',
                         'StartTimeC',
                         'EndTimeC',
                         'Duration',
                         'Notes']
                for idx, name in enumerate(names):
                    dct[name] = entry[idx]
                self.items.append(dct)
        print "Spreadsheet loaded.", len(self.items), "entries."

    @staticmethod
    def video_matches_entry(entry, (hive, tagc, beeid, videoid)):
        """ Check if a video matches an entry in the spreadsheet. """
        return entry['VideoID'] == videoid and \
               entry['TagColor'] == tagc and \
               entry['TagNumber'] == beeid and \
               entry['Hive'] == hive

    def get_video_events(self, params):
        """ Return a list of dictionaries, where all dict items belong to video videoid. """
        return [entry for entry in self.items if self.video_matches_entry(entry, params)]

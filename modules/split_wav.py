from pydub import AudioSegment
import math

class SplitWavAudioMubin():
    def __init__(self, folder, filename, target_folder):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.target_folder = target_folder
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.target_folder + '/' + split_filename, format="wav")
        
    def multiple_split(self, start_sec=0, sec_per_split=3, overlap=0):
        total_sec = math.ceil(self.get_duration())
        for i in range(start_sec, total_sec, int(sec_per_split-overlap)):
            split_fn = self.filename.replace('.wav', '') + '_{}.wav'.format(i) # saves file as dir/filename_i.wav
            self.single_split(i, i+sec_per_split, split_fn)
            print(str(i) + ' Done')
        print('All splited successfully')



import argparse
import os
import scaper
import numpy as np
import predict



""" Constansts for Scaper """
# OUTPUT FOLDER
OUTFOLDER = "C:/Soundscape"
# SCAPER SETTINGS
FG_FOLDER = "C:/Soundbank/foreground"
BG_FOLDER = "C:/Soundbank/background"
N_SOUNDSCAPES = 1
REF_DB = -50
DURATION = 30.0

MIN_EVENTS = 3
MAX_EVENTS = 9

EVENT_TIME_DIST = 'truncnorm'
EVENT_TIME_MEAN = 5.0
EVENT_TIME_STD = 2.0
EVENT_TIME_MIN = 0.0
EVENT_TIME_MAX = 10.0

SOURCE_TIME_DIST = 'const'
SOURCE_TIME = 0.0

EVENT_DURATION_DIST = 'uniform'
EVENT_DURATION_MIN = 7
EVENT_DURATION_MAX = 12

SNR_DIST = 'uniform'
SNR_MIN = 6
SNR_MAX = 30

PITCH_DIST = 'uniform'
PITCH_MIN = -1.0
PITCH_MAX = 1.0

TIME_STRETCH_DIST = 'uniform'
TIME_STRETCH_MIN = 0.8
TIME_STRETCH_MAX = 1.2
# generate a random seed for this Scaper object
SEED = 123

class SoundGenerator():

    def __init__(self, foreground_sounds, background_sounds, image_names):
        # Initialisation of Scaper and Object-Detection Container
        self.sc = scaper.Scaper(DURATION, FG_FOLDER, BG_FOLDER, random_state=SEED)
        self.sc.protected_labels = []
        self.sc.ref_db = REF_DB
        self.detected_foreground_sounds = foreground_sounds
        self.detected_background_sounds = background_sounds 
        self.image_names = image_names

    # Generate 2 soundscapes using a truncated normal distribution of start times
    def generate_sound(self, n_soundscapes):
        for i in range(len(self.image_names)):
            image_name = self.image_names[i]
            fg_sound = self.detected_foreground_sounds[i]
            bg_sound = self.detected_background_sounds[i]
            all_foreground_sounds_list = os.listdir(FG_FOLDER)
            all_background_sounds_list = os.listdir(BG_FOLDER)
            final_fg_sound = [x for x in fg_sound if x in all_foreground_sounds_list]
            final_bg_sound = [x for x in bg_sound if x in all_background_sounds_list]

            for n in range(len(n_soundscapes)):

                print('Generating soundscape: {:d}/{:d}'.format(n+1, len(n_soundscapes)))

                # reset the event specifications for foreground and background at the
                # beginning of each loop to clear all previously added events
                self.sc.reset_bg_event_spec()
                self.sc.reset_fg_event_spec()

                # add background
                self.sc.add_background(label=('choose', final_bg_sound),
                                source_file=('choose', []),
                                source_time=('const', 0))

                # add random number of foreground events
                n_events = np.random.randint(MIN_EVENTS, MAX_EVENTS+1)
                for _ in range(n_events):
                    self.sc.add_event(label=('choose', final_fg_sound),
                                source_file=('choose', []),
                                source_time=(SOURCE_TIME_DIST, SOURCE_TIME),
                                event_time=(EVENT_TIME_DIST, EVENT_TIME_MEAN, EVENT_TIME_STD, EVENT_TIME_MIN, EVENT_TIME_MAX),
                                event_duration=(EVENT_DURATION_DIST, EVENT_DURATION_MIN, EVENT_DURATION_MAX),
                                snr=(SNR_DIST, SNR_MIN, SNR_MAX),
                                pitch_shift=(PITCH_DIST, PITCH_MIN, PITCH_MAX),
                                time_stretch=(TIME_STRETCH_DIST, TIME_STRETCH_MIN, TIME_STRETCH_MAX))

                # generate
                audiofile = os.path.join(OUTFOLDER, "{}{:d}.wav".format(image_name,n))
                jamsfile = os.path.join(OUTFOLDER, "{}{:d}.jams".format(image_name,n))
                txtfile = os.path.join(OUTFOLDER, "{}{:d}.txt".format(image_name,n))

                self.sc.generate(audiofile, jamsfile,
                            allow_repeated_label=True,
                            allow_repeated_source=True,
                            reverb=0.1,
                            disable_sox_warnings=True,
                            no_audio=False,
                            txt_path=txtfile)
                
                print("Path to output folder: {}".format(OUTFOLDER))
                        
if __name__ == "__main__":
    os.chdir("extended_prototype")
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_height', type=int, default=512, help='Image height after resizing')
    parser.add_argument('--img_width', type=int, default=1024, help='Image width after resizing')
    parser.add_argument('--weights', type=str, default="pretrained/pretrained.h5", help='Relative path of network weights')

    args = parser.parse_args()
    foreground_objects, background_objects = predict.main(args)
    image_names = list(foreground_objects.keys())
    foreground_objects_list = list(foreground_objects.values())
    background_objects_list = list(background_objects.values())
    soundscape_generator = SoundGenerator(foreground_sounds=foreground_objects_list, background_sounds=background_objects_list, image_names=image_names)
    number_soundscapes_per_image = range(0, 3)
    soundscape_generator.generate_sound(number_soundscapes_per_image)

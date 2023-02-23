import json
import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument('--transcript_dir', dest='transcript_dir',
                    type=str, help='Directory for transcripts')
parser.add_argument('--motion_dir', dest='transcript_dir',
                    type=str, help='Directory for transcripts')
args = parser.parse_args()

motion_dir = args.motion_dir
transcript_dir = args.transcript_dir

recording_files = [f for f in listdir(motion_dir) if isfile(join(motion_dir, f))]

transcript = json.load(args.transcript_dir)

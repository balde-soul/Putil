# coding=utf-8
import os
import re
import subprocess
import sys

def is_fsynth_installed():
    """ Check to make sure fluidsynth exists in the PATH """
    for path in os.environ['PATH'].split(os.pathsep):
        f = os.path.join(path, 'fluidsynth')
        if os.path.exists(f) and os.access(f, os.X_OK):
            return True
    return False

def to_audio(sf2, midi_file, out_dir, out_type='wav', txt_file=None, append=True):
    """ 
    Convert a single midi file to an audio file.  If a text file is specified,
    the first line of text in the file will be used in the name of the output
    audio file.  For example, with a MIDI file named '01.mid' and a text file
    with 'A    major', the output audio file would be 'A_major_01.wav'.  If
    append is false, the output name will just use the text (e.g. 'A_major.wav')
    
    Args:
        sf2 (str):        the file path for a .sf2 soundfont file
        midi_file (str):  the file path for the .mid midi file to convert
        out_dir (str):    the directory path for where to write the audio out
        out_type (str):   the output audio type (see 'fluidsynth -T help' for options)
        txt_file (str):   optional text file with additional information of how to name 
                          the output file
        append (bool):    whether or not to append the optional text to the original
                          .mid file name or replace it
    """
    fbase = os.path.splitext(os.path.basename(midi_file))[0]
    if not txt_file:
        out_file = out_dir + '/' + fbase + '.' + out_type
    else:
        line = 'out'
        with open(txt_file, 'r') as f:
            line = re.sub(r'\s', '_', f.readline().strip())
            
        if append:
            out_file = out_dir + '/' + line + '_' + fbase + '.' + out_type
        else:
            out_file = out_dir + '/' + line + '.' + out_type

    subprocess.call(['fluidsynth', '-F', out_file, sf2, midi_file])

def main():
    """
    Convert a directory of MIDI files to audio files using the following command line options:
    
    --sf2-dir (required)   the path to a directory with .sf2 soundfont files.  The script will 
                           pick a random soundfont from this directory for each file.
                           
    --midi-dir (required)  the path to a directory with the .mid MIDI files to convert.
    
    --out-dir (optional)   the directory to write the audio files to
    
    --type (optional)      the audio type to write out (see 'fluidsynth -T help' for options)
                           the default is 'wav'
                           
    --replace (optional)   if .txt files exist in the same directory as the .mid files, the text
                           from the files will be used for the output audio file names instead
                           of the midi file names.  If not specified, the text from the files will
                           be appended to the file name.
    """
    import argparse
    import random
    import glob

    parser = argparse.ArgumentParser()
    UbuntuSf2DefaultPath = '/usr/share/sounds/sf2'
    parser.add_argument('--sf2_dir', dest='SF2Dir', type=str, default=UbuntuSf2DefaultPath, action='store', help='the path to a directory with \
        .sf2 soundfont files. \n The script wil pick a random soundfont from this directory for each file \
        the default path in \n ubuntu is {0}'.format(UbuntuSf2DefaultPath))
    parser.add_argument('--midi_dir', dest='MidiDir', type=str, default='', action='store', help='the path to a directory \
        with .mid MIDI files to convert')
    parser.add_argument('--out_dir', dest='OutDir', type=str, default='', action='store', help='the directory to write the audio files to')
    out_type_default = 'wav'
    parser.add_argument('--out_type', dest='OutType', type=str, default='wav', action='store', help='the audio type to write out \
        (see \'fluidsynth -T help\' for options the default is {0}'.format(out_type_default))
    parser.add_argument('--replace', dest='Replace', default=False, action='store_true', help='if .txt files exist in the same directory \
        as the .mid files, \n the text from the files will be used for the output audio file name instead of the midi files names. \
        if not specified, \n the text from the files will be appended to the file name')
    try:
        if not is_fsynth_installed():
            raise Exception('Unable to find \'fluidsynth\' in the path')
        
        options = parser.parse_args()
        sf2files, midifiles, textfiles, out_dir, out_type, append = [], [], [], None, 'wav', True
        sf2files = glob.glob(options.SF2Dir + '/*.[sS][fF]2')
        midifiles = glob.glob(options.MidiDir + '/*.[mM][iI][dD]')
        textfiles = glob.glob(options.MidiDir + '/*.[tT][xX][tT]')
        out_dir = options.OutDir
        out_type = options.OutType
        append = not options.Replace
        if not sf2files:
            raise Exception('A --sf2-dir directory must be specified where at least one .sf2 file exists')
        elif not midifiles:
            raise Exception('A --midi-dir directory must be specified where at least one .mid file exists')

        if not textfiles or len(textfiles) < len(midifiles):
            for mid in midifiles:
                to_audio(random.choice(sf2files), mid, out_dir, out_type)
        else:
            for mid, txt in zip(midifiles, textfiles):
                to_audio(random.choice(sf2files), mid, out_dir, out_type, txt, append)
                    
    except Exception as exc:
        print(str(exc))
        sys.exit(2)
        
if __name__ == '__main__':
    sys.exit(main())
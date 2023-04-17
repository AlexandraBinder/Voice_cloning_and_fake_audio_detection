import speech_recognition as sr
import os
import glob
import wave

import argparse
import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from sphfile import SPHFile

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

def convert_audiofiles(root_path):
    """
    Converts .WAV files with NIST_1 headers into .wav files with RIFF headers
    and stores the new version in the same directory.

    :param root_path: root path where audiofiles are stored
    """
    for root, dirs, files in os.walk(root_path, topdown=False):
        for file in files:
            _, file_extension = os.path.splitext(file)
            if file_extension == '.WAV':
                file_path = os.path.join(root, file)
                sph = SPHFile(file_path)
                txt_file = ""
                txt_file = file.split('.')[0] + ".TXT"
                f = open(os.path.join(root, txt_file),'r')
                
                for line in f:
                    words = line.split(" ")
                    start_time = (int(words[0])/16000)
                    end_time = (int(words[1])/16000)

                sph.write_wav(file_path.replace(".WAV",".wav"),start_time,end_time) 
            
        if root == root_path: break

def organize_audio_files(root_path, new_directory):
    """
    Copies .wav files stored in root_path and organizes them into a new directory.

    :param root_path: root path where audiofiles are stored
    :param new_directory: new folder where audiofiles will be stored
    """
    for root, dirs, files in os.walk(root_path, topdown=False):
        aux = '_'.join(root.split('/')[-2:])
        for file in files:
            _, file_extension = os.path.splitext(file)
            if file_extension == '.wav':
                src = os.path.join(root, file)
                dst = os.path.join(new_directory, '_'.join([aux, file]))
                shutil.copyfile(src, dst)
        if root == root_path: break

def check_cuda():
    """
    Checks CUDA CPU and or GPU information
    """
    new_directory
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

import numpy as np

def wer(reference, hypothesis):
    """
    Calculates the Word Error Rate (WER) between two sentences.

    :param r: reference sentence
    :param h: hypothesis sentence
    :return: the WER between the reference and hypothesis sentences
    """
    # split sentences into words
    reference = reference.split()
    hypothesis = hypothesis.split()

    # create a matrix with the size of (len(r)+1)x(len(h)+1)
    matrix = np.zeros((len(reference)+1)*(len(hypothesis)+1), dtype=np.uint16)
    matrix = matrix.reshape((len(reference)+1, len(hypothesis)+1))

    # initialize the first row and column of the matrix
    for i in range(len(reference)+1):
        for j in range(len(hypothesis)+1):
            if i == 0:
                matrix[0][j] = j
            elif j == 0:
                matrix[i][0] = i

    # fill the matrix
    for i in range(1, len(reference)+1):
        for j in range(1, len(hypothesis)+1):
            if r[i-1] == hypothesis[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                substitution = matrix[i-1][j-1] + 1
                insertion = matrix[i][j-1] + 1
                deletion = matrix[i-1][j] + 1
                matrix[i][j] = min(substitution, insertion, deletion)

    # the WER is the last element of the matrix
    return matrix[len(reference)][len(hypothesis)]/len(reference)

def voice_to_text(voice):
    """
    Transcribes an audio file into text

    :param voice: source audio file to transcribe.
    :return: text transcription of audio file
    """
    r = sr.Recognizer()
    with sr.AudioFile(str(voice)) as src:
        r.adjust_for_ambient_noise(src)
        audio = r.listen(src)
        try:
            text = r.recognize_google(audio, language='en-in')
            return text
        except Exception:
            print(f"Couldn't understand {voice} file")

def text_to_voice(sample_voice, audio_text, dest_path):
    """
    Generates a fake audio file by doing a text to voice conversion.
    The fake audio is stored in a specific directory.

    :param sample_voice: audio file to use as a sample for the text to voice conversion.
    :param audio_text: text to generate fake audio file
    :param dst_audio: path to where the fake audio file will be stored.
    """
    try:
        original_wav, sampling_rate = librosa.load(str(sample_voice))
        preprocessed_wav = encoder.preprocess_wav(original_wav)

        # Since the synthesizer works in batch, 
        # the data needs to be put in a list or numpy array
        embed = encoder.embed_utterance(preprocessed_wav)
        embeds = [embed]
        texts = [audio_text]

        synthesizer = Synthesizer("saved_models/default/synthesizer.pt");
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]

        generated_wav = vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)

        sf.write(dest_path, generated_wav.astype(np.float32), synthesizer.sample_rate)

    except Exception as e:
        print("Caught Exception: %s" % repr(e))

def voice_cloning(src_audio, sample_voice, dst_audio):
    """
    Clones the voice of a given audio file. 
    First it transcribes a source audio into text and then it generates a 
    fake audio file by doing a text to voice conversion.
    The fake audio is stored in a specific directory.

    :param src_audio: source audio file to clone.
    :param sample_voice: audio file to use as a sample for the text to voice conversion.
    :param dst_audio: path to where the fake audio file will be stored.
    """
    text = voice_to_text(src_audio)
    text_to_voice(sample_voice, text, dst_audio)
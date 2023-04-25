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
from jiwer import wer

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical

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

def organize_audio_files(root_path, new_wav_folder, new_txt_folder):
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
                dst = os.path.join(new_wav_folder, '_'.join([aux, file]))
                shutil.copyfile(src, dst)
            elif file_extension == '.TXT':
                src = os.path.join(root, file)
                dst = os.path.join(new_txt_folder, '_'.join([aux, file]))
                shutil.copyfile(src, dst)
        if root == root_path: break

def check_cuda():
    """
    Checks CUDA CPU and or GPU information
    """
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

def get_original_text(file):
    with open(file, 'r') as f:
        data = f.read()
    return ' '.join(data.split(' ')[2:])

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

def voice_cloning(src_audio, src_text_file, sample_voice, dst_audio):
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
    src_text = get_original_text(src_text_file)
    word_error_rate = wer(reference=src_text, hypothesis=text)
    text_to_voice(sample_voice, text, dst_audio)
    return word_error_rate

def calculate_MFCCs_and_labels(files_path, y_value):
    """
    Calculates the Mel-frequency cepstral coefficients (MFCCs) of the 
    audio files stored in the files_path and then pads the resulting 
    MFCC matrix with zeros to have a fixed shape of (n, 40). 
    It then return the padded MFCC matrices and the correspondig labels 
    (0 for original, 1 for fake)

    :param files_path: path to audio files
    :y_value: label value to return
    :return: returns a list X of the padded MFCC matrices and a list y of their corresponding labels
    """
    file_path, X, y = [], [], []
    for file in os.listdir(files_path):
        wav, _ = librosa.load(os.path.join(files_path , file))
        mfcc = librosa.feature.mfcc(wav)
        padded_mfcc = pad2d(mfcc, 40)
        file_path.append(file)
        X.append(padded_mfcc)
        y.append(y_value)
    return file_path, X, y

def pad2d(mfcc, i):
    """
    Pads a MFCC matrix with zeros to have a fixed shape of (1, i). 

    :param files_path: path to audio files
    :y_value: label value to return
    :return: returns a padded MFCC matrix
    """
    if mfcc.shape[1] > i: return mfcc[:, 0: i]
    else: return np.hstack((mfcc, np.zeros((mfcc.shape[0], i - mfcc.shape[1]))))

def get_train_test_split(X, y, test_size, val_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=val_size, random_state=42)
    
    X_train, X_val, X_test, y_train, y_val, y_test = map(np.array, [X_train, X_val, X_test, y_train, y_val, y_test])

    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)

    return X_train, X_val, X_test, y_train, y_val, y_test
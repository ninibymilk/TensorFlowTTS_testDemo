import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import pyaudio
import wave
import soundfile as sf

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

tacotron2_config = AutoConfig.from_pretrained('TensorFlowTTS/examples/tacotron2/conf/tacotron2.baker.v1.yaml')
tacotron2 = TFAutoModel.from_pretrained(
    config=tacotron2_config,
    pretrained_path="tacotron2-100000.h5",
    name="tacotron2"
)

mb_melgan_config = AutoConfig.from_pretrained(
    'TensorFlowTTS/examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml')
mb_melgan = TFAutoModel.from_pretrained(
    config=mb_melgan_config,
    pretrained_path="mb.melgan.h5",
    name="mb_melgan"
)

processor = AutoProcessor.from_pretrained(pretrained_path="./baker_mapper.json")

# 合成
def do_synthesis(input_text, text2mel_model, vocoder_model):
    input_ids = processor.text_to_sequence(input_text, inference=True)

    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )

    remove_end = 1024
    audio = vocoder_model.inference(mel_outputs)[0, :-remove_end, 0]
    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()


def visualize_attention(alignment_history):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f'Alignment steps')
    im = ax.imshow(
        alignment_history,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_mel_spectrogram(mels):
    mels = tf.reshape(mels, [-1, 80]).numpy()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(311)
    ax1.set_title(f'Predicted Mel-after-Spectrogram')
    im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
    plt.show()
    plt.close()


def play(f):
    chunk = 1024
    wf = wave.open(f, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
    data = wf.readframes(chunk)
    while data != b'':
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()


input_text = "这是一个开源的端到端中文语音合成系统"

tacotron2.setup_window(win_front=5, win_back=5)

mels, alignment_history, audios = do_synthesis(input_text, tacotron2, mb_melgan)
visualize_attention(alignment_history[0])
visualize_mel_spectrogram(mels[0])
sf.write('demo_cn.wav', audios, 24000)
play('demo_cn.wav')

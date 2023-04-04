import streamlit as st
from pathlib import Path
# import time
# import glob
# import os
import soundfile as sf
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

temp = Path('temp')
temp.mkdir(exist_ok=True, parents=True)

print(SpectrogramGenerator.list_available_models())
# Download and load the pretrained tacotron2 model@
@st.cache_resource
def load_model(spec_generator, vocoder):
    generator = SpectrogramGenerator.from_pretrained(spec_generator)
    # Download and load the pretrained waveglow model
    vocode = Vocoder.from_pretrained(vocoder)

    return generator, vocode

spectrum_generator, vocoder = load_model('tts_en_tacotron2', 'tts_hifigan')
st.title("Text to speech")
text = st.text_input("Enter text")

def text_to_speech(text):
   # All spectrogram generators start by parsing raw strings to a tokenized version of the string
    parsed = spectrum_generator.parse(str(text))
    # They then take the tokenized string and produce a spectrogram
    spectrogram = spectrum_generator.generate_spectrogram(tokens=parsed,)
    # Finally, a vocoder converts the spectrogram to audio
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    sf.write(f"temp/{my_file_name}.mp3", audio.to('cpu').detach().numpy()[0], 22050)
    return my_file_name

if st.button("convert"):
    result = text_to_speech(text)
    audio_file = open(f"temp/{result}.mp3", "rb")
    audio_bytes = audio_file.read()
    st.markdown(f"## Your audio:")
    st.audio(audio_bytes, format="audio/mp3", start_time=0)



# def remove_files(n):
#     mp3_files = glob.glob("temp/*mp3")
#     if len(mp3_files) != 0:
#         now = time.time()
#         n_days = n * 86400
#         for f in mp3_files:
#             if os.stat(f).st_mtime < now - n_days:
#                 os.remove(f)
#                 print("Deleted ", f)


# remove_files(7)
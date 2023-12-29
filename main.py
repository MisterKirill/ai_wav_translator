from os import path, makedirs
from TTS.api import TTS
import glob
import speech_recognition as sr
from googletrans import Translator

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
SRC_LANG = 'en'
DEST_LANG = 'ru'

tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')

translator = Translator()


def main():
    files = glob.glob(path.join(INPUT_DIR, '*/*.wav'))
    for i, file in enumerate(files):
        output_file = file.replace(INPUT_DIR, OUTPUT_DIR)

        r = sr.Recognizer()
        with sr.AudioFile(file) as source:
            audio = r.record(source)

        try:
            text_transcribed = r.recognize_google(audio)
        except sr.UnknownValueError:
            try:
                text_transcribed = r.recognize_sphinx(audio)
            except sr.UnknownValueError:
                print(f'{file}: failed')
                continue

        text_translated = translator.translate(text_transcribed, src=SRC_LANG, dest=DEST_LANG).text

        makedirs(path.dirname(output_file), exist_ok=True)

        try:
            tts.tts_to_file(
                text=text_translated,
                file_path=output_file,
                speaker_wav=file,
                language=DEST_LANG
            )
        except RuntimeError:
            print(f'{file} ({i + 1}/{len(files)} {100 * float(i + 1) / float(len(files))}%): failed')
            continue

        print(f'{file} ({i + 1}/{len(files)} {100 * float(i + 1) / float(len(files))}%): {text_transcribed} -> {text_translated}')


if __name__ == '__main__':
    main()

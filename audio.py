import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import sys
import select

r = sr.Recognizer()

def esperar_enter():
    print("Presiona ENTER para empezar a grabar...")
    input()
    print("Grabando... Presiona ENTER nuevamente para detener.")

def enter_presionado():
    # Revisa si ENTER fue presionado sin bloquear
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def grabar_audio(nombre="temp.wav", fs=16000):
    esperar_enter()

    recording = sd.InputStream(samplerate=fs, channels=1, dtype='int16')
    audio_data = []

    with recording:
        while True:
            if enter_presionado():  # si presionan ENTER, detener
                sys.stdin.readline()  # limpiar buffer
                break

            audio_data.append(recording.read(1024)[0])

    data = np.concatenate(audio_data, axis=0)
    wav.write(nombre, fs, data)
    print("Grabación detenida y guardada en", nombre)


# Grabar indefinidamente hasta que se presione ENTER nuevamente
grabar_audio()

# --- Transcribir ---
with sr.AudioFile("temp.wav") as source:
    audio = r.record(source)

texto = r.recognize_google(audio, language="es-ES")
print("Texto transcrito:", texto)
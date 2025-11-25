import { useState, useRef } from "react";
import "./App.css";

function App() {
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [transcription, setTranscription] = useState("");

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);

    audioChunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      audioChunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      const formData = new FormData();
      formData.append("audio_file", blob, "audio.wav");

      try {
        const res = await fetch("http://127.0.0.1:8000/api/transcribe", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        setTranscription(data.transcription || "");
      } catch (err) {
        setTranscription("Error al transcribir: " + err.message);
      }
    };

    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", textAlign: "center" }}>
      <h1>Grabador + Transcripción</h1>

      {!recording ? (
        <button onClick={startRecording}>🎙️ Iniciar grabación</button>
      ) : (
        <button onClick={stopRecording}>⏹️ Detener grabación</button>
      )}

      {audioUrl && (
        <div>
          <h3>Audio grabado</h3>
          <audio controls src={audioUrl} />
        </div>
      )}

      <h3>Transcripción</h3>
      <textarea
        value={transcription}
        onChange={(e) => setTranscription(e.target.value)}
        style={{ width: "100%", height: "180px" }}
      />

      <button
        style={{ marginTop: "20px" }}
        onClick={() => console.log("Enviar al modelo:", transcription)}
      >
        Enviar al modelo
      </button>
    </div>
  );
}

export default App;

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
import os
import tempfile
from pydub import AudioSegment
import boto3
import json

app = FastAPI(
    title="Audio AI API",
    version="2.0.0",
    description="API simplificada para transcripción de audio y predicción con SageMaker"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar reconocedor de voz
recognizer = sr.Recognizer()

# Cliente de SageMaker
sagemaker_runtime = None
try:
    sagemaker_runtime = boto3.client(
        'sagemaker-runtime',
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    print("✓ SageMaker client inicializado correctamente")
except Exception as e:
    print(f"⚠ SageMaker client no disponible: {e}")


# Modelo para el request del segundo endpoint
class TextInput(BaseModel):
    text: str
    endpoint_name: str = None


@app.get("/")
async def root():
    """Health check básico"""
    return {
        "status": "online",
        "service": "Audio AI API",
        "version": "2.0.0",
        "sagemaker_available": sagemaker_runtime is not None
    }


@app.post("/api/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint para transcribir audio a texto
    
    Args:
        audio_file: Archivo de audio (wav, mp3, ogg, etc.)
    
    Returns:
        JSON con el texto transcrito
    """
    try:
        # Validar tipo de archivo
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser de tipo audio"
            )
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio_file.read()
            temp_audio.write(content)
            temp_path = temp_audio.name
        
        try:
            # Convertir a WAV si es necesario
            audio = AudioSegment.from_file(temp_path)
            wav_path = temp_path.replace(temp_path.split('.')[-1], 'wav')
            audio.export(wav_path, format="wav")
            
            # Transcribir con Google Speech Recognition
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                texto = recognizer.recognize_google(audio_data, language="es-ES")
            
            # Limpiar archivos temporales
            os.unlink(temp_path)
            if wav_path != temp_path:
                os.unlink(wav_path)
            
            return {
                "success": True,
                "transcription": texto,
                "filename": audio_file.filename
            }
            
        except sr.UnknownValueError:
            raise HTTPException(
                status_code=400,
                detail="No se pudo entender el audio"
            )
        except sr.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error en el servicio de transcripción: {str(e)}"
            )
        finally:
            # Asegurar limpieza de archivos
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando el audio: {str(e)}"
        )




@app.post("/api/text-to-model")
async def text_to_model(input_data: TextInput):
    """
    ENDPOINT 2: Recibe JSON con texto, se conecta a SageMaker, envía el texto
    y devuelve la respuesta del modelo
    """
    if not sagemaker_runtime:
        raise HTTPException(
            status_code=503,
            detail="SageMaker no está configurado. Verifica las credenciales de AWS."
        )
    
    try:
        # Obtener nombre del endpoint
        endpoint = input_data.endpoint_name or os.getenv('SAGEMAKER_ENDPOINT_NAME')
        if not endpoint:
            raise HTTPException(
                status_code=400,
                detail="Debes proporcionar 'endpoint_name' o configurar SAGEMAKER_ENDPOINT_NAME en .env"
            )
        
        # Preparar payload para SageMaker
        payload = {
            "text": input_data.text,
            "language": "es"
        }
        
        # Invocar endpoint de SageMaker
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Leer y parsear respuesta
        result = json.loads(response['Body'].read().decode())
        
        return {
            "model_response": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando el texto: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🎤 Audio AI API - Iniciando servidor")
    print("="*50)
    print(f"📍 Servidor: http://0.0.0.0:8000")
    print(f"📖 Docs: http://0.0.0.0:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
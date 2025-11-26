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

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Este es el texto que se enviará al modelo",
                "endpoint_name": "mi-endpoint-sagemaker"
            }
        }


@app.get("/")
async def root():
    """Health check básico"""
    return {
        "status": "online",
        "service": "Audio AI API",
        "version": "2.0.0",
        "sagemaker_available": sagemaker_runtime is not None
    }


@app.post("/api/audio-to-text")
async def audio_to_text(audio_file: UploadFile = File(...)):
    """
    ENDPOINT 1: Recibe audio, lo transcribe y devuelve JSON con el texto
    
    Args:
        audio_file: Archivo de audio (wav, mp3, ogg, etc.)
    
    Returns:
        {
            "text": "texto transcrito del audio"
        }
    """
    temp_path = None
    wav_path = None
    
    try:
        # Validar que sea un archivo de audio
        if audio_file.content_type and not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser de tipo audio"
            )
        
        # Guardar audio en archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_audio:
            content = await audio_file.read()
            temp_audio.write(content)
            temp_path = temp_audio.name
        
        # Convertir a WAV (formato compatible con speech_recognition)
        audio = AudioSegment.from_file(temp_path)
        wav_path = temp_path.replace('.tmp', '.wav')
        audio.export(wav_path, format="wav")
        
        # Transcribir audio
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            texto = recognizer.recognize_google(audio_data, language="es-ES")
        
        return {
            "text": texto
        }
        
    except sr.UnknownValueError:
        raise HTTPException(
            status_code=400,
            detail="No se pudo entender el audio. Asegúrate de que contenga voz clara."
        )
    except sr.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error en el servicio de transcripción: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando el audio: {str(e)}"
        )
    finally:
        # Limpiar archivos temporales
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


@app.post("/api/text-to-model")
async def text_to_model(input_data: TextInput):
    """
    ENDPOINT 2: Recibe JSON con texto, se conecta a SageMaker, envía el texto
    y devuelve la respuesta del modelo
    
    Args:
        input_data: {
            "text": "texto a procesar",
            "endpoint_name": "nombre-del-endpoint" (opcional)
        }
    
    Returns:
        {
            "model_response": "respuesta del modelo de IA"
        }
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
        
    except boto3.exceptions.Boto3Error as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error conectando con SageMaker: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando el texto: {str(e)}"
        )


@app.post("/api/audio-to-model")
async def audio_to_model(audio_file: UploadFile = File(...)):
    """
    ENDPOINT 3: FLUJO COMPLETO - Audio directo al modelo
    
    Diseñado especialmente para ESP32 y aplicaciones que necesitan
    respuesta directa sin pasos intermedios.
    
    Flujo:
    1. Recibe audio
    2. Transcribe automáticamente
    3. Envía al modelo de IA
    4. Devuelve respuesta completa + versión corta para ESP32
    
    Args:
        audio_file: Archivo de audio (wav, mp3, ogg, etc.)
    
    Returns:
        {
            "success": true,
            "transcription": "texto transcrito",
            "model_response": {...},  # Respuesta completa del modelo
            "short_response": "...",   # Versión corta (max 200 chars) para ESP32
            "filename": "audio.wav"
        }
    """
    temp_path = None
    wav_path = None
    
    try:
        # ========== PASO 1: TRANSCRIBIR AUDIO ==========
        print(f"📝 Procesando audio: {audio_file.filename}")
        
        # Validar tipo de archivo
        if audio_file.content_type and not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser de tipo audio"
            )
        
        # Guardar audio temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_audio:
            content = await audio_file.read()
            temp_audio.write(content)
            temp_path = temp_audio.name
        
        # Convertir a WAV
        audio = AudioSegment.from_file(temp_path)
        wav_path = temp_path.replace('.tmp', '.wav')
        audio.export(wav_path, format="wav")
        
        # Transcribir
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            texto = recognizer.recognize_google(audio_data, language="es-ES")
        
        print(f"✓ Transcripción: {texto[:80]}...")
        
        # ========== PASO 2: ENVIAR AL MODELO ==========
        if not sagemaker_runtime:
            # Sin SageMaker: devolver solo transcripción
            short_text = texto if len(texto) <= 200 else texto[:197] + "..."
            return {
                "success": True,
                "transcription": texto,
                "model_response": None,
                "short_response": short_text,
                "filename": audio_file.filename,
                "note": "SageMaker no configurado. Solo transcripción disponible."
            }
        
        # Obtener endpoint
        endpoint = os.getenv('SAGEMAKER_ENDPOINT_NAME')
        if not endpoint:
            short_text = texto if len(texto) <= 200 else texto[:197] + "..."
            return {
                "success": True,
                "transcription": texto,
                "model_response": None,
                "short_response": short_text,
                "filename": audio_file.filename,
                "note": "Endpoint de SageMaker no configurado."
            }
        
        # Preparar payload
        payload = {
            "text": texto,
            "language": "es"
        }
        
        # Invocar SageMaker
        print(f"🤖 Enviando al modelo: {endpoint}")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parsear respuesta
        model_result = json.loads(response['Body'].read().decode())
        print(f"✓ Respuesta del modelo recibida")
        
        # ========== PASO 3: CREAR VERSIÓN CORTA PARA ESP32 ==========
        short_text = ""
        
        # Intentar extraer el texto más relevante
        if isinstance(model_result, dict):
            # Buscar campos comunes de respuesta
            for key in ['response', 'answer', 'result', 'text', 'output', 'prediction']:
                if key in model_result:
                    short_text = str(model_result[key])
                    break
            
            # Si no encontramos, usar el primer valor
            if not short_text and model_result:
                short_text = str(list(model_result.values())[0])
        else:
            short_text = str(model_result)
        
        # Limitar a 200 caracteres para displays pequeños (ESP32)
        if len(short_text) > 200:
            short_response = short_text[:197] + "..."
        else:
            short_response = short_text
        
        return {
            "success": True,
            "transcription": texto,
            "model_response": model_result,
            "short_response": short_response,
            "filename": audio_file.filename
        }
        
    except sr.UnknownValueError:
        raise HTTPException(
            status_code=400,
            detail="No se pudo entender el audio. Asegúrate de que contenga voz clara."
        )
    except sr.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error en el servicio de transcripción: {str(e)}"
        )
    except boto3.exceptions.Boto3Error as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error conectando con SageMaker: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en el flujo completo: {str(e)}"
        )
    finally:
        # Limpiar archivos temporales
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🎤 Audio AI API - Iniciando servidor")
    print("="*50)
    print(f"📍 Servidor: http://0.0.0.0:8000")
    print(f"📖 Docs: http://0.0.0.0:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
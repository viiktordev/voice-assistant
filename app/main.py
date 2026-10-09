import asyncio
import functools
import math
import io
import json
import os
import queue
import re
import subprocess
import wave
from array import array
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import riva.client
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

ASR_SERVER = os.getenv('ASR_SERVER', 'host.docker.internal:50051')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
TTS_URL = os.getenv('TTS_URL', 'http://host.docker.internal:8000/synthesize')
TTS_STREAM_URL = os.getenv('TTS_STREAM_URL', TTS_URL.rstrip('/') + '/stream')  # PCM 16 bits cru em pedaços
ASR_LANGUAGE = os.getenv('ASR_LANGUAGE', 'pt-BR')
TTS_LANGUAGE = os.getenv('TTS_LANGUAGE', 'Portuguese')
TTS_SPEAKER = os.getenv('TTS_SPEAKER', 'Ryan')
ASR_END_SILENCE = int(os.getenv('ASR_END_SILENCE_MS', '800')) / 1000  # silêncio (energia baixa) que encerra a fala
ASR_VAD_MIN_RMS = int(os.getenv('ASR_VAD_MIN_RMS', '50'))  # energia mínima (PCM 16 bits) para contar como voz; acima disso vale 3x o ruído de fundo
ASR_TEXT_STALL = 2.5        # reserva para ambiente ruidoso: transcrição parada por esse tempo também encerra
ASR_NO_SPEECH_TIMEOUT = 8   # segundos ouvindo sem reconhecer nada
ASR_MAX_SECONDS = 60
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'Você é um assistente de voz útil. Responda em português brasileiro de modo natural, claro e conciso. Evite markdown, listas longas e símbolos que ficam estranhos quando falados.')

app = FastAPI(title='Local Voice Assistant')
STATIC = Path(__file__).parent / 'static'

class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    history: list[dict[str, str]] = Field(default_factory=list)

class StreamChatRequest(ChatRequest):
    speak: bool = True

class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

@app.get('/')
async def index():
    return FileResponse(STATIC / 'index.html')

@app.get('/health')
async def health():
    return {'status': 'ok', 'asr_server': ASR_SERVER, 'ollama_model': OLLAMA_MODEL, 'tts_language': TTS_LANGUAGE}


def decode_audio_to_pcm(data: bytes) -> bytes:
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('Audio maior que o limite de 16 MB')
    proc = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'], input=data, capture_output=True, timeout=45)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError('Não foi possível decodificar áudio: ' + proc.stderr.decode(errors='replace')[:350])
    return proc.stdout


@functools.cache
def asr_service() -> riva.client.ASRService:
    return riva.client.ASRService(riva.client.Auth(uri=ASR_SERVER))


def recognition_config() -> riva.client.RecognitionConfig:
    return riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        audio_channel_count=1,
        language_code=ASR_LANGUAGE,
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )


def recognize(data: bytes) -> str:
    pcm = decode_audio_to_pcm(data)
    response = asr_service().offline_recognize(pcm, recognition_config())
    return ' '.join(result.alternatives[0].transcript for result in response.results if result.alternatives).strip()

@app.post('/api/transcribe')
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read(16 * 1024 * 1024 + 1)
    try:
        transcript = await asyncio.to_thread(recognize, data)
        return {'text': transcript}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'Falha no STT: {exc}') from exc

@app.websocket('/ws/listen')
async def listen(ws: WebSocket):
    """Cliente envia frames binários com PCM 16 kHz mono s16le enquanto a pessoa fala; servidor responde
    {type: partial} com a transcrição parcial, {type: end_of_speech} ao detectar o fim da fala (o cliente
    para o microfone) e {type: final} com o texto pontuado. O cliente pode forçar o fim com {type: stop}.

    O servidor ASR não implementa endpointing (ignora endpointing_config e só envia is_final quando o
    áudio acaba), então o fim da fala é detectado aqui: depois de reconhecer alguma palavra, energia do
    áudio abaixo do limiar por ASR_END_SILENCE ou transcrição parada por ASR_TEXT_STALL. Só a transcrição
    não basta: após uma pausa o modelo demora ~0,8 s para emitir a próxima palavra."""
    await ws.accept()
    loop = asyncio.get_running_loop()
    audio: queue.Queue[bytes | None] = queue.Queue()
    events: asyncio.Queue[tuple[bool, str] | Exception | None] = asyncio.Queue()

    def audio_chunks():
        while (chunk := audio.get()) is not None:
            yield chunk

    # O cliente Riva é síncrono: roda numa thread e devolve os resultados ao event loop.
    def run_asr():
        try:
            config = riva.client.StreamingRecognitionConfig(config=recognition_config(), interim_results=True)
            for response in asr_service().streaming_response_generator(audio_chunks(), config):
                for result in response.results:
                    if result.alternatives:
                        loop.call_soon_threadsafe(events.put_nowait, (result.is_final, result.alternatives[0].transcript.strip()))
        except Exception as exc:
            loop.call_soon_threadsafe(events.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(events.put_nowait, None)

    last_voice = noise_floor = None

    async def receive_audio():
        nonlocal last_voice, noise_floor
        while True:
            message = await ws.receive()
            if message['type'] == 'websocket.disconnect':
                break
            if chunk := message.get('bytes'):
                audio.put(chunk)
                samples = array('h', chunk[:len(chunk) // 2 * 2])
                if samples:
                    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
                    if rms >= 1:  # blocos de zeros (microfone abrindo) zerariam o piso de ruído
                        noise_floor = rms if noise_floor is None else min(noise_floor, rms)
                    if rms > max(ASR_VAD_MIN_RMS, 3 * (noise_floor or 0)):
                        last_voice = loop.time()
            elif message.get('text') and json.loads(message['text']).get('type') == 'stop':
                break
        audio.put(None)

    asr_task = asyncio.create_task(asyncio.to_thread(run_asr))  # referência evita coleta da task
    receiver = asyncio.create_task(receive_audio())
    finals, partial = [], ''

    def apply(event: tuple[bool, str]) -> bool:
        nonlocal partial
        is_final, text = event
        if is_final:
            if text:
                finals.append(text)
            partial = ''
            return True
        changed = text != partial
        partial = text
        return changed

    try:
        started = last_change = loop.time()
        stream_ended = False
        while True:
            now = loop.time()
            if (finals or partial) and last_voice is not None:
                deadline = min(last_change + ASR_TEXT_STALL, max(last_voice, last_change) + ASR_END_SILENCE)
            elif finals or partial:
                deadline = last_change + ASR_TEXT_STALL  # energia nunca passou do limiar (microfone baixo): só pelo texto
            else:
                deadline = started + ASR_NO_SPEECH_TIMEOUT
            deadline = min(deadline, started + ASR_MAX_SECONDS)
            if now >= deadline:
                break  # fim da fala, ninguém falou ou limite de duração
            try:
                # Acorda no máximo a cada 100 ms: last_voice avança no receive_audio sem gerar evento.
                event = await asyncio.wait_for(events.get(), min(deadline - now, 0.1))
            except TimeoutError:
                continue
            if event is None:
                stream_ended = True  # cliente mandou stop e o ASR já devolveu tudo
                break
            if isinstance(event, Exception):
                raise event
            if apply(event):
                last_change = loop.time()
                await ws.send_json({'type': 'partial', 'text': ' '.join([*finals, partial]).strip()})

        if not stream_ended:
            await ws.send_json({'type': 'end_of_speech'})
            audio.put(None)  # fecha o stream; o ASR devolve o resultado final pontuado
            while (event := await asyncio.wait_for(events.get(), 15)) is not None:
                if isinstance(event, Exception):
                    raise event
                apply(event)
        await ws.send_json({'type': 'final', 'text': ' '.join(finals) or partial})
    except Exception as exc:
        try:
            await ws.send_json({'type': 'error', 'detail': f'Falha no STT: {exc}'})
        except Exception:
            pass  # cliente já desconectou
    finally:
        audio.put(None)
        receiver.cancel()  # a thread do ASR termina sozinha ao receber o None
        try:
            await ws.close()
        except Exception:
            pass

def build_messages(request: ChatRequest) -> list[dict[str, str]]:
    history = [m for m in request.history[-12:] if m.get('role') in ('user', 'assistant') and isinstance(m.get('content'), str)]
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, *history, {'role': 'user', 'content': request.text}]


def ollama_payload(messages: list[dict[str, str]], stream: bool) -> dict:
    return {'model': OLLAMA_MODEL, 'stream': stream, 'think': False, 'messages': messages, 'options': {'num_ctx': 4096}}


async def stream_ollama(client: httpx.AsyncClient, messages: list[dict[str, str]]):
    async with client.stream('POST', f'{OLLAMA_URL.rstrip("/")}/api/chat', json=ollama_payload(messages, stream=True)) as result:
        if result.is_error:
            raise RuntimeError(f'HTTP {result.status_code}: {(await result.aread()).decode(errors="replace")[:350]}')
        async for line in result.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if 'error' in chunk:
                raise RuntimeError(chunk['error'])
            if text := chunk.get('message', {}).get('content'):
                yield text


async def synthesize_speech(client: httpx.AsyncClient, text: str) -> bytes:
    result = await client.post(TTS_URL, json={'text': text, 'language': TTS_LANGUAGE, 'speaker': TTS_SPEAKER})
    result.raise_for_status()
    content_type = result.headers.get('content-type', 'audio/wav')
    if not content_type.startswith('audio/') and not result.content.startswith(b'RIFF'):
        raise ValueError(f'Resposta não é áudio: {result.text[:350]}')
    return result.content


async def stream_speech(client: httpx.AsyncClient, text: str):
    """Gera (sample_rate, pedaço PCM s16le) conforme o TTS produz o áudio. Agrupa em pedaços de ~100 ms
    com número par de bytes, já que a rede pode partir uma amostra de 16 bits ao meio."""
    async with client.stream('POST', TTS_STREAM_URL, json={'text': text, 'language': TTS_LANGUAGE, 'speaker': TTS_SPEAKER}) as result:
        if result.is_error:
            raise RuntimeError(f'HTTP {result.status_code}: {(await result.aread()).decode(errors="replace")[:350]}')
        sample_rate = int(result.headers.get('x-sample-rate', '24000'))
        min_bytes = sample_rate // 10 * 2
        pending = b''
        async for data in result.aiter_bytes():
            pending += data
            if len(pending) >= min_bytes:
                cut = len(pending) // 2 * 2
                yield sample_rate, pending[:cut]
                pending = pending[cut:]
        if len(pending) >= 2:
            yield sample_rate, pending[:len(pending) // 2 * 2]


# Fim de frase: pontuação seguida de espaço (ignora "3.5") ou quebra de linha.
SENTENCE_END = re.compile(r'[.!?…]+["\')\]]*\s+|\n+')
MIN_SENTENCE_CHARS = 12   # evita cortar em "Sr. " ou mandar fragmentos minúsculos ao TTS
MAX_SENTENCE_CHARS = 250  # sem pontuação por muito tempo, corta numa vírgula ou espaço


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Separa as frases completas do buffer e devolve (frases, resto)."""
    sentences, start = [], 0
    for match in SENTENCE_END.finditer(buffer):
        candidate = buffer[start:match.end()].strip()
        if len(candidate) >= MIN_SENTENCE_CHARS:
            sentences.append(candidate)
            start = match.end()
    rest = buffer[start:]
    while len(rest) > MAX_SENTENCE_CHARS:
        cut = rest.rfind(',', 0, MAX_SENTENCE_CHARS) + 1
        if cut < MAX_SENTENCE_CHARS // 2:
            cut = rest.rfind(' ', 0, MAX_SENTENCE_CHARS)
        if cut <= 0:
            cut = MAX_SENTENCE_CHARS
        sentences.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    return [s for s in sentences if s], rest

@app.post('/api/chat')
async def chat(request: ChatRequest):
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            result = await client.post(f'{OLLAMA_URL.rstrip("/")}/api/chat', json=ollama_payload(build_messages(request), stream=False))
            result.raise_for_status()
            answer = result.json()['message']['content'].strip()
            return {'text': answer}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'Falha no Ollama: {exc}') from exc

@app.post('/api/synthesize')
async def synthesize(request: SpeechRequest):
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            return Response(await synthesize_speech(client, request.text), media_type='audio/wav')
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'Falha no TTS: {exc}') from exc

@app.websocket('/ws/chat')
async def chat_stream(ws: WebSocket):
    """Um turno por conexão. Cliente envia {text, history, speak}; servidor responde com
    {type: delta} a cada pedaço do LLM, {type: audio_format, sample_rate} antes do primeiro áudio, frames
    binários com PCM s16le mono conforme o TTS gera cada frase, {type: text} com a resposta completa e
    {type: done} ao final. Fechar o socket cancela o turno."""
    await ws.accept()
    send_lock = asyncio.Lock()

    async def send(data: dict | bytes):
        async with send_lock:
            if isinstance(data, bytes):
                await ws.send_bytes(data)
            else:
                await ws.send_json(data)

    worker = None
    try:
        request = StreamChatRequest.model_validate(await ws.receive_json())
        async with httpx.AsyncClient(timeout=180) as client:
            sentences: asyncio.Queue[str | None] = asyncio.Queue()

            # Sintetiza em ordem, uma frase por vez, enquanto o LLM continua gerando.
            async def tts_worker():
                sample_rate = None
                while (sentence := await sentences.get()) is not None:
                    try:
                        async for rate, pcm in stream_speech(client, sentence):
                            if rate != sample_rate:
                                sample_rate = rate
                                await send({'type': 'audio_format', 'sample_rate': rate})
                            await send(pcm)
                    except Exception as exc:
                        raise RuntimeError(f'Falha no TTS: {exc}') from exc

            if request.speak:
                worker = asyncio.create_task(tts_worker())

            answer, buffer = '', ''
            try:
                async for delta in stream_ollama(client, build_messages(request)):
                    answer += delta
                    await send({'type': 'delta', 'text': delta})
                    if worker and not worker.done():  # se o TTS falhou, o texto segue e o erro sai no await abaixo
                        buffer += delta
                        ready, buffer = split_sentences(buffer)
                        for sentence in ready:
                            sentences.put_nowait(sentence)
            except Exception as exc:
                raise RuntimeError(f'Falha no Ollama: {exc}') from exc

            await send({'type': 'text', 'text': answer.strip()})
            if worker:
                if buffer.strip():
                    sentences.put_nowait(buffer.strip())
                sentences.put_nowait(None)
                await worker
        await send({'type': 'done'})
    except Exception as exc:
        try:
            await send({'type': 'error', 'detail': str(exc)})
        except Exception:
            pass  # cliente já desconectou (cancelamento)
    finally:
        if worker and not worker.done():
            worker.cancel()
        try:
            await ws.close()
        except Exception:
            pass

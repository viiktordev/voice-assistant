"""Verifica se o servidor Riva/Nemotron aceita reconhecimento offline e streaming.

Uso no host da GPU, com o container rodando:
    docker exec -i voice-assistant python - < scripts/check_asr.py
"""
import os

import grpc
import riva.client
from riva.client.proto import riva_asr_pb2

server = os.getenv('ASR_SERVER', 'localhost:50051')
language = os.getenv('ASR_LANGUAGE', 'pt-BR')
asr = riva.client.ASRService(riva.client.Auth(uri=server))
print(f'Servidor: {server}  idioma: {language}\n')

print('== Modelos carregados')
try:
    response = asr.stub.GetRivaSpeechRecognitionConfig(riva_asr_pb2.RivaSpeechRecognitionConfigRequest())
    for model in response.model_config:
        params = dict(model.parameters)
        interesting = {k: v for k, v in params.items() if k in ('type', 'streaming', 'offline', 'language_code', 'sample_rate')}
        print(f'- {model.model_name}: {interesting or params}')
except grpc.RpcError as exc:
    print(f'Não foi possível listar ({exc.code().name}: {exc.details()})')

pcm = b'\0\0' * 16000 * 2  # 2 s de silêncio, PCM 16 kHz mono
config = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,
    audio_channel_count=1,
    language_code=language,
    max_alternatives=1,
    enable_automatic_punctuation=True,
)

print('\n== Offline (Recognize)')
try:
    asr.offline_recognize(pcm, config)
    print('OK')
except grpc.RpcError as exc:
    print(f'FALHOU ({exc.code().name}: {exc.details()})')

print('\n== Streaming (StreamingRecognize)')
try:
    streaming_config = riva.client.StreamingRecognitionConfig(config=config, interim_results=True)
    chunks = [pcm[i:i + 3200] for i in range(0, len(pcm), 3200)]  # blocos de 100 ms
    for _ in asr.streaming_response_generator(audio_chunks=chunks, streaming_config=streaming_config):
        pass
    print('OK')
except grpc.RpcError as exc:
    print(f'FALHOU ({exc.code().name}: {exc.details()})')

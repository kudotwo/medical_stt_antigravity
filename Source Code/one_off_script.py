import sys, json
# sys.path.insert(0, 'Source Code')   # adjust if you run this from elsewhere
import stt_pipeline_gpu as stt

# reuse the cached segments -- no need to re-transcribe audio
with open('audio_sample/audio_recordings/Audio_Recordings/CAR0001_segments.json') as f:
    segments = json.load(f)

result = stt.diarize_and_extract_soap(segments)   # uses your updated prompt + GPT-5.6

print(json.dumps(result, indent=2, ensure_ascii=False))
curl - X POST "http://192.168.0.102:8000/v1/audio/speech" \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "input": "这件事让 Peter 非常震惊。",
    "instructions": "读到'非常'这个词时，语速放慢，语气加重，带有明显的停顿和强调感",
    "response_format": "wav"
  }' -o output.wav

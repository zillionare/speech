curl -X POST http://localhost:8123/v1/audio/speech \                                                            
    -H "Authorization: Bearer 1234" \                                                                             
    -H "Content-Type: application/json" \                                                                         
    -d '{                                                                                                         
      "model": "VibeVoice-Realtime-0.5B-8bit",                                                                    
      "input": "你好，我是 Aaron",                                                                                
      "voice": "Bowen",                                                                                           
      "response_format": "wav"                                                                                    
    }' \                                                                                                          
    --output output.wav \                                                                                         
    --max-time 300 

import sys
sys.path.pop(0)
import whisper

model = whisper.load_model("base")
result = model.transcribe(audio="audios/1_Installing VS Code & How Websites Work.mp3",
                          language="hi",
                          task="translate")

print(result["text"])
 
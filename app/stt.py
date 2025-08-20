import whisper

def transcribe_audio(audio_file: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

if __name__ == "__main__":
    print(transcribe_audio("data/samples/sample1.wav"))

from google.cloud import speech_v1
import io


def sample_recognize(local_file_path, model):
    """
    Transcribe a short audio file using a specified transcription model

    Args:
      local_file_path Path to local audio file, e.g. /path/audio.wav
      model The transcription model to use, e.g. video, phone_call, default
      For a list of available transcription models, see:
      https://cloud.google.com/speech-to-text/docs/transcription-model#transcription_models
    """

    client = speech_v1.SpeechClient()

    # local_file_path = 'resources/hello.wav'
    # model = 'phone_call'

    # The language of the supplied audio
    language_code = "ms-MY"#malay language code
    config = {"model": model, "language_code": language_code}
    with io.open(local_file_path, "rb") as f:
        content = f.read()
    audio = {"content": content}

    response = client.recognize(config, audio)
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))
if __name__ == '__main__':
    sample_recognize(r'C:\Users\vamsi\Downloads\Life-changing-Motivational- audio-by-ujjwal-patni.wav','phone_call')
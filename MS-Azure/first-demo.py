import azure.cognitiveservices.speech as speechsdk

speech_key, service_region = "", "germanywestcentral"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

speech_config.speech_synthesis_voice_name = "de-DE-GiselaNeural"

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# Receives a text from console input.
print("Type some text that you want to speak...")
text = input()

result = speech_synthesizer.speak_text_async(text).get()

# Checks result
import azure.cognitiveservices.speech as speechsdk

while True:
    user_input = input("Geben Sie 'x' ein, um die Schleife abzubrechen: ")

    if user_input == 'x':
        break

    # Fügen Sie hier den Code für die Sprachsynthese ein
    # Die Bedingungen für die Schleife können innerhalb dieser if-Anweisungen geändert werden
    elif result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))

    # Eine Meldung, die bei jeder Iteration der Schleife ausgegeben wird
    print("Did you update the subscription info?")

    

import pyttsx3

engine = pyttsx3.init()
engine.say("I will speak this text")
engine.runAndWait()
#
# engine= pyttsx3.init()
# engine.setProperty('rate',70)
# voices=engine.getProperty('voices')
# for voice in voices:
#     print("Using voice:"), repr(voice)
#     engine.setProperty('voice',voice.id)
#     engine.say("Hello Hello Hello")
# engine.runAndWait()

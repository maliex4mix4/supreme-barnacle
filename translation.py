"""
    Translation module
"""
import os
import time
import threading

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()

TRANSLATION_TEMPLATE = """
Translate the following sentence into {language}, return ONLY the translation, nothing else.

Sentence: {sentence}
"""

output_parser = StrOutputParser()
llm = ChatOpenAI(temperature=0.0, model="gpt-4")
translation_prompt = ChatPromptTemplate.from_template(TRANSLATION_TEMPLATE)

translation_chain = (
  {"language": RunnablePassthrough(), "sentence": RunnablePassthrough()}
  | translation_prompt
  | llm
  | output_parser
)

def translate(sentence, language):
    """
        Translate the given sentence into the specified language.
    """
    data_input = {"language": language, "sentence": sentence}
    translation = translation_chain.invoke(data_input)
    return translation

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def gen_dub(text):
    """
        Generate dubbing audio for the given text.
    """
    st.write("Generating audio...")
    audio = client.generate(
        text=text,
        voice="Adam",
        model="eleven_multilingual_v2"
    )
    # Convert the generator to bytes
    audio_bytes = b''.join(audio)
    st.audio(audio_bytes, format="audio/mp3")
    play(audio_bytes)

def on_data(transcript: aai.RealtimeTranscript, language):
    """
        Callback function to handle real-time transcription data.
    """
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        st.write(transcript.text)
        st.write("Translating...")
        start_time = time.time()
        translation = translate(str(transcript.text), language)
        st.write(f"Translation: {translation}")
        gen_dub(translation)
        end_time = time.time()
        st.write(f"\n\n{'+'*40}\nTime taken: {end_time - start_time:.2f} seconds\n{'+'*40}\n")
    else:
        st.write(transcript.text)

def on_error(error: aai.RealtimeError):
    """
        Callback function to handle transcription errors.
    """
    st.error(f"An error occurred: {error}")

def start_transcription(language):
    """
        Start real-time transcription.
    """
    transcriber = aai.RealtimeTranscriber(
        on_data=lambda transcript: on_data(transcript, language),
        on_error=on_error,
        sample_rate=44_100,
    )

    transcriber.connect()
    microphone_stream = aai.extras.MicrophoneStream()

    try:
        transcriber.stream(microphone_stream)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        transcriber.close()
        st.write("Stopped listening.")

def main():
    """
        Main function to run the Streamlit app.
    """
    st.title("Real-time Translation and Dubbing")

    language = st.selectbox("Select target language", ["Turkish", "Spanish", "French", "German"])

    if st.button("Start Translation"):
        st.write("Listening... Speak now.")
        thread = threading.Thread(target=start_transcription, args=(language,))
        thread.start()

if __name__ == "__main__":
    main()

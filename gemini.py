from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
def text_to_speech():
        from gtts import gTTS
        import os
        import tempfile
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


        tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")


        def translate_text(text, target_language="en"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)


            translated_ids = model.generate(input_ids, attention_mask=attention_mask, num_beams=4, early_stopping=True, forced_bos_token_id=tokenizer.lang_code_to_id[target_language])


            translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
            return translated_text


        print("Enter some text that you want to translate and speak >")
        text =response

        translated_text = translate_text(text)
        print("Translated Text:", translated_text)


        tts = gTTS(text=translated_text, lang='en', slow=False)


        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            temp_audio_file.close()
            tts.save(temp_audio_file.name)


        #audio_file_path = os.system(f"start {temp_audio_file.name}")
        audio_file_path = temp_audio_file.name
        st.audio(audio_file_path)



def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load OpenAI model and get respones

def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

##initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")

st.header("Gemini Application")

input=st.text_input("Input: ",key="input")


submit=st.button("Ask the question")

## If ask button is clicked

if submit:
    global response
    response=get_gemini_response(input)
    st.subheader("The Response is")

    st.write(response)
    text_to_speech()




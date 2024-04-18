import streamlit as st
st.title("AI FOR YOU...")
question = st.text_input("How May I Help You, Please Ask Question ?")
pr =st.button("Submit")

# import streamlit as st
# import pyttsx3
# import speech_recognition as sr
# from PIL import Image



# def speech_to_text():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         recognizer.adjust_for_ambient_noise(source)
#         # st.image(Image.open("listening.png"), use_column_width=True)
#         audio = recognizer.listen(source)
        
            

#         try:
#             text = recognizer.recognize_google(audio)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand audio"
#         except sr.RequestError as e:
#             return f"Error: {e}"

# def main():
#     st.title("Speech Recognition with Streamlit")

#     # Speech to text
#     st.subheader("Speech to Text")
#     st.write("Click the button and speak into the microphone:")
#     if st.button("Start Recording 🎙️"):
#         text = speech_to_text()   
#         st.write("You said:", text)

# if __name__ == "__main__":
#     main()

def text_to_answer():
        
        
        import langchain

        # import langsmith

        from langchain.llms import OpenAI

        from langchain.agents import load_agent,initialize_agent,AgentType

        import os
        os.environ["OPEN_API_KEY"]=str("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")

        os.environ['SERPAPI_API_KEY']=str("fe1c8fefa12050e7194ca4ecaa21c4fe568bdf67eaf6627396fd18b125b9145f")

        from uuid import uuid4

        unique_id = uuid4().hex[0:8]

        os.environ["LANGCHAIN_TRACING_V2"]="true"

        # os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"

        os.environ["LANGCHAIN_PROJECT"]="MRKL"

        os.environ["LANGCHAIN_API_KEY"] ="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"
        from langchain.agents import ZeroShotAgent,Tool,AgentExecutor

        from langchain import OpenAI,SerpAPIWrapper,LLMChain

        search = langchain.SerpAPIWrapper()

        

        tools = [

            Tool(

                name="search",

                func=search.run,

                description="useful for when you need to answer questions about current events",

            )

        ]
        prefix = """Answer the following questions as best you can. You have access to the following tools:"""

        suffix = """When answering, you MUST speak in the following language: {language}.

        

        Question: {input}

        {agent_scratchpad}"""

        

        prompt = ZeroShotAgent.create_prompt(

            tools,

            prefix=prefix,

            suffix=suffix,

            input_variables=["input", "language", "agent_scratchpad"],

        )

        

        llm_chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key="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"), prompt=prompt)

        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)

        

        agent_executor = AgentExecutor.from_agent_and_tools(

            agent=agent, tools=tools, verbose=True

        )
        print(prompt.template)
        

        output = agent_executor.run({"input": question,"language": "en"})
        return output

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
        text =text_to_answer() 

        translated_text = translate_text(text)
        print("Translated Text:", translated_text)


        tts = gTTS(text=translated_text, lang='en', slow=False)


        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            temp_audio_file.close()
            tts.save(temp_audio_file.name)


        #audio_file_path = os.system(f"start {temp_audio_file.name}")
        audio_file_path = temp_audio_file.name
        st.audio(audio_file_path)

if pr:
    # st.write(question)

    text_to_speech()





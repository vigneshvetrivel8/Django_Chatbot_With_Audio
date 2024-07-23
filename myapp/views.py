from django.shortcuts import render
import pinecone
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
import io
import speech_recognition as sr
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pydub import AudioSegment

# Initialize Pinecone index
index = pinecone.Index(host="<HOST-NAME>", api_key="<API-KEY>")

import os
os.environ["GOOGLE_API_KEY"] = "<GOOGLE-GEMINI-API>"

# Initialize generative AI and embeddings models
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize conversation memory
memory = ConversationBufferMemory()

# Initialize ConversationChain with language model and memory
conversation = ConversationChain(llm=llm, memory=memory)

def get_response(user_query):
    query_embedding = embeddings_model.embed_query(user_query)

    # Perform a similarity search in the Pinecone index
    matching_ids = index.query(vector=query_embedding, top_k=3)
    content_list = []

    for id in matching_ids['matches']:
        result = index.fetch(ids=[id['id']])
        content = result['vectors'][str(id['id'])]['metadata']['content']
        content_list.append(content)

    prompt = (
        """
        Question: {question}
        Context: {context}

        Rules:
        1) Answer the question according to the context provided.
        2) If context is related but doesn't provide clarity or enough information, ask the user to contact the business directly.
        3) If the question is too general or unrelated to the context or something like greetings, answer directly.
        """
    ).format(context=content_list, question=user_query)

    # Predict response based on prompt
    output = conversation.predict(input=prompt)
    return output


@csrf_exempt
def audio_to_text_view(request):
    if request.method == 'POST':
        audio_file = request.FILES['audio']

        # Convert audio file to WAV format using pydub
        try:
            audio = AudioSegment.from_file(audio_file)
            audio = audio.set_channels(1)  # Convert to mono
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
        except Exception as e:
            return JsonResponse({'error': f'Error converting audio file: {str(e)}'}, status=400)

        # Use SpeechRecognition to recognize the text
        recognizer = sr.Recognizer()
        audio_data = sr.AudioFile(wav_io)

        try:
            with audio_data as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_text = recognizer.record(source)

            # Recognize the text using Google Cloud Speech API
            try:
                text = recognizer.recognize_google(audio_text)
            except sr.UnknownValueError:
                text = "Sorry, I could not understand the audio."
            except sr.RequestError:
                text = "Sorry, there was an error with the speech recognition service."

            return JsonResponse({'text': text})

        except Exception as e:
            return JsonResponse({'error': f'Error processing audio file: {str(e)}'}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def index_view(request):
    answer = None
    if request.method == 'POST':
        question = request.POST.get('question')
        if question:
            answer = get_response(question)
    return render(request, 'myapp/index.html', {'answer': answer})

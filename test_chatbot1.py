import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import langchain
import base64
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit_javascript import st_javascript
import google.generativeai as genai
import os
from langchain.schema import HumanMessage, AIMessage
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from location import read_gendarmeries_database, calculate_distance, generate_google_maps_link, get_user_location, find_nearest_station, generate_openstreetmap_image_url
from langdetect import detect
from pdf2image import convert_from_path
from pytesseract import image_to_string
import PyPDF2

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        return "unknown"

from googletrans import Translator


logo = r"C:\Users\user\Documents\Stages\ChatBot\data\images\genai1.png"  # Remplacez par le chemin de votre image

st.logo(logo, icon_image=logo)




translator = Translator()

def translate_to_french(text):
    return translator.translate(text, src='auto', dest='fr').text

def translate_to_original_lang(text, target_lang):
    return translator.translate(text, src='fr', dest=target_lang).text


gen_config=genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Récupérer la clé API et l'ID du moteur de recherche personnalisé
api_key = os.getenv("SEARCH_API_KEY")
cx = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

def extract_text_from_scanned_pdf(pdf_path):
    text = ""
    # Convertir chaque page du PDF en image
    images = convert_from_path(pdf_path)
    for image in images:
        # Utiliser pytesseract pour extraire le texte de l'image
        text += image_to_string(image, lang='fra')  # Utiliser 'fra' pour le français
    return text

def extract_text_from_pdf(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def extract_text_from_any_pdf(pdf_path):
    try:
        # Essayer d'extraire le texte directement
        text = extract_text_from_pdf(pdf_path)
        if text.strip():  # Si le texte est trouvé
            return text
    except Exception as e:
        print(f"Error reading PDF with PyPDF2: {e}")
    
    # Utiliser l'OCR comme solution de secours
    print("Trying OCR...")
    return extract_text_from_scanned_pdf(pdf_path)



def get_text_chunks(text):
    text_splitter= CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        
    )
    chunks= text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore=FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        generation_config=gen_config,
        messages=[
        {
            "role": "system",
            "content": (
                "Tu es un assistant conversationnel lié à la Brigade Nationale"
                "de Veille d'Alerte et d'Assistance (BNVAA). Ton rôle est de répondre de manière précise et "
                "détaillée aux questions des utilisateurs en utilisant le contexte fourni ou les"
                "informations que tu peux obtenir."
                "Si tu ne trouves pas de réponse dans le contexte donné, dis : "
                "'La réponse n'est pas accessible à partir du contexte donné.' "
                "Ensuite, effectue une recherche web ou utilise des sources externes pour aider l'utilisateur."
            )
        }
        ]
    )
    
    memory=ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    
    
    conversation_chain= ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def google_search(query, api_key, cse_id, num_results=3):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = [
            {"title": item['title'], "url": item['link']}  # Modifié pour renvoyer un dictionnaire
            for item in data.get("items", [])
        ]
        return results
    else:
        return [{"title": "Erreur", "url": f"Erreur : {response.status_code} - {response.text}"}]  # Renvoi d'un dictionnaire en cas d'erreur


def display_chat():
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            avatar = "🤖" if isinstance(message, AIMessage) else "😎"
            content = message.content
            with st.container():
                if isinstance(message, HumanMessage):
                    st.markdown(
                        f"<div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>"
                        f"<div style='background-color: #F0F0F0;color: #333333;border-radius: 10px;padding: 10px;max-width: 60%'>"
                        f"<strong>{avatar}</strong> {content}</div></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>"
                        f"<div style='background-color: #E0F7FA;color: #00796B;border-radius: 10px;padding: 10px;max-width: 60%;'>"
                        f"<strong>{avatar}</strong> {content}</div></div>",
                        unsafe_allow_html=True
                    )
    

    
    
               

def handle_userinput(user_question):
    # Détecter la langue de la question
    detected_lang = detect_language(user_question)
    # Ajouter la question de l'utilisateur sous forme d'objet HumanMessage
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    # Traiter la question dans la langue détectée
    if detected_lang != "fr":  # Si la langue n'est pas le français
        st.session_state.chat_history.append(AIMessage(content="Je détecte que vous avez posé une question dans une autre langue. Laissez-moi traduire ou répondre si possible."))
        # Vous pouvez utiliser une API ou un outil de traduction ici pour adapter le texte si nécessaire
        user_question_translated = translate_to_french(user_question)  # Exemple d'appel à une fonction de traduction
    else:
        user_question_translated = user_question
    
    response = st.session_state.conversation({
        "question": user_question_translated,
        "chat_history": st.session_state.chat_history  # Passer uniquement les objets HumanMessage/AIMessage
    })
    
   
    
    # Ajouter la réponse de l'assistant sous forme d'objet AIMessage
    assistant_response = response.get("answer")
    #st.write(assistant_response)
    if ("Je ne sais pas" in assistant_response or 
        "Pas de réponse" in assistant_response or 
        "Je suis désolé" in assistant_response):
        assistant_response = (
            "La réponse n'est pas accessible à partir du contexte donné. "
            "Je vais effectuer une recherche web pour essayer de vous répondre."
        )
        # Insérer ici la logique pour effectuer une recherche web
        # Exemple :
        web_results = google_search(user_question, 
                                api_key, 
                                cx)
        if web_results:
            extracted_texts = []
            for result in web_results:
                url = result['url']
                
                # text_from_url=extract_text_from_url(url)
                # extracted_texts.append(text_from_url)
                try:
                    text_from_url = extract_text_from_url(url)
                    extracted_texts.append(text_from_url)
                except Exception:
                    # Ignorer les erreurs et passer au site suivant
                    continue
            # Combiner les textes extraits et afficher comme réponse
            assistant_response += "\n\nVoici la réponse basée sur les résultats de la recherche :\n" + "\n".join(extracted_texts)
     
            
            #formatted_results = [(f"[{result['title']}]({result['url']})" for result in web_results)]
            formatted_results = [
                f"- [{result['title']}]" for result in web_results
            ]
            # for result in web_results:
            #     assistant_response += f"    ** {result['title']}\n\n"
            assistant_response += "Liens de recherche :\n" + "\n".join(formatted_results)
            #handle_search_results(web_results)
        else:
            assistant_response += "\nAucun résultat pertinent trouvé."
        # search_results = web.search(user_question)
        # assistant_response += f"\nRésultat de recherche : {search_results}"
    else:
        assistant_response=response.get("answer")
        
    
    if detected_lang != "fr":
        assistant_response = translate_to_original_lang(assistant_response, detected_lang)  # Traduire la réponse vers la langue d'origine
   
    st.session_state.chat_history.append(AIMessage(content=assistant_response))
    st.session_state.user_question = ""
    
    #st.write(response)  # Cela affiche le contenu de la réponse dans Streamlit.
      # Container for chat messages
   
            
def handle_search_results(search_results):
    # Récupérer et afficher les résultats de recherche sous forme de liens cliquables
    if search_results:
        st.write("Voici les résultats de la recherche :")
        for result in search_results:
            title = result.get('title', 'Pas de titre')
            url = result.get('url', '#')
            st.markdown(f"- [{title}]({url})")
 

@st.dialog("Formulaire de plainte")
def plainte(item):
    st.write("Remplissez le formulaire")
    st.write("Lien pour le formulaire pour vol: https://docs.google.com/forms/d/e/1FAIpQLSdHbnhJA4uCfuI5imDVINTkzbV-uj1_kahzDmNLgnSGm7MCZw/viewform?vc=0&c=0&w=1&flr=0&usp=mail_form_link")
    st.write("Lien pour le formulaire concernant l'agression: https://docs.google.com/forms/d/e/1FAIpQLSd5_E2-Sf4k6rcVr9I-K4JS2WbN8WMxIhwHhqU6cS4Fcizlqw/viewform?vc=0&c=0&w=1&flr=0&usp=mail_form_link")
    st.write("Lien pour le formulaire concernant les litiges pour les propriétés:https://docs.google.com/forms/d/181xFKBmxzDQjlgpwqi2euV7r5B4TeedFkVLzIh3omPM/edit")
    st.write("Lien pour le formulaire concernant les accidents de circulation: https://docs.google.com/forms/d/1T22fNY6O7fhoZlXrXwERwDSKRxHgZ0s29tQ3YKv4BQQ/edit?usp=forms_home&ths=true")
   


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie si la requête a réussi
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Optionnel : Enregistrer le HTML modifié par BeautifulSoup
        with open("page_modifiee.html", "w", encoding="utf-8") as file:
            file.write(soup.prettify())  # Prettify formatte le HTML pour une meilleure lisibilité

        # Extraire un contenu pertinent (par exemple, tout le texte du body)
        paragraphs = soup.find_all('p')  # Récupérer tous les paragraphes
        text = ' '.join([para.get_text() for para in paragraphs])
        # Vérifiez si le texte est suffisant pour un résumé
        if not text.strip() or len(text.split()) < 50:
            return "Le contenu de la page est trop court pour être résumé."
        
        if text:
             
            max_input_length = 300
            truncated_text = ' '.join(text.split()[:max_input_length])
            # Utiliser le modèle de résumé pour créer un résumé court
            summary = summarizer(truncated_text, max_length=70, min_length=50, do_sample=False)
            return summary[0]['summary_text']  # Retourne le résumé généré
        else:
            return "Aucune information disponible"
        # Limiter la taille de la réponse pour éviter des réponses trop longues
        return ' '.join(text.split()[:300])  # Retourner les 300 premiers mots pour un résumé
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte: {str(e)}")
        return ""



def get_pdfs_from_folder(folder_path):
        pdf_files = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pdf'):  # Vérifie que le fichier est un PDF
                pdf_files.append(os.path.join(folder_path, file_name))
        return pdf_files
    
folder_path = r"C:\Users\user\Documents\Stages\ChatBot\data\data"

# Charger le modèle de résumé
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def extract_summary_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie si la requête a réussi
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extraire le texte de tous les paragraphes de la page
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs if para.get_text()])
        # Vérifiez si le texte est suffisant pour un résumé
        if not text.strip() or len(text.split()) < 50:
            return "Le contenu de la page est trop court pour être résumé."
       

        if text:
            # Utiliser le modèle de résumé pour créer un résumé court
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
            return summary[0]['summary_text']  # Retourne le résumé généré
        else:
            return "Aucune information disponible"
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte: {str(e)}")
        return ""

def scroll_to_bottom():
    st_javascript("""
    window.scrollTo(0, document.body.scrollHeight);
    """)   
 

def main():
    load_dotenv()
    st.set_page_config(page_title="My GenAIBot", page_icon="🤖",layout="centered", initial_sidebar_state="auto")
    st.header("Discutez avec Maya GendAI 👮‍♀️🇧🇫")
    st.text("Maya, votre protectrice digitale")
    
    # Ajouter une image à gauche dans la barre latérale
    sidebar_image_path = r"C:\Users\user\Documents\Stages\ChatBot\data\images\im1.jpg"  # Remplacez par le chemin de votre image
    if os.path.exists(sidebar_image_path):
        # sidebar_image = Image.open(sidebar_image_path)
        # st.sidebar.image(sidebar_image, use_column_width=True)
        # Convertir l'image en base64
        with open(sidebar_image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        # HTML pour centrer l'image
        st.sidebar.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{encoded_string}" alt="Image Barre Latérale" style="width: 100px; height: auto; border-radius: 10px;">
            </div>
            """,
            unsafe_allow_html=True
        ) # Ajouter une image au centre avant le titre
    center_image_path = r"C:\Users\user\Documents\Stages\ChatBot\data\images\im1bg.png"  # Remplacez par le chemin de votre image
   
    if os.path.exists(center_image_path):
        # Charger l'image
        #center_image = Image.open(center_image_path)

        # Convertir l'image en base64
        with open(center_image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        # HTML pour centrer l'image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{encoded_string}" alt="Bienvenue !" style="width: 300px; height: auto; border-radius: 10px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("L'image spécifiée n'existe pas.")
        # Initialiser l'état de session pour stocker la conversation
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # Stocke la chaîne
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Stocke l'historique des messages
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""  # Stocke la question de l'utilisateur

    # Container for user input
   
    
    
    # input_container = st.container()
    
    # with input_container:
    #     #display_chat()
    # # Champ de saisie placé en bas
        
    #     st.markdown("""<div id="input-container" style='height: 20px;'></div>""", unsafe_allow_html=True)  # Espacement pour séparer les messages et l'entrée
        
    #     user_question= st.text_input("Posez votre question: ", value=st.session_state.user_question, key="input_field")
    #     if user_question:
    #         handle_userinput(user_question)
    #         display_chat()
    #         st.session_state.user_question = ""
    #         scroll_to_bottom()
    #         st.markdown(
    #             """
    #             <script>
    #                 document.getElementById('input-container').scrollIntoView({ behavior: 'smooth' });
    #             </script>
    #             """,
    #             unsafe_allow_html=True
    #         )
            
    #     #user_question=""
        
   
    
    gendarmeries = read_gendarmeries_database()
    with st.sidebar:
        st.subheader("Menu")
        if os.path.exists(folder_path):        
                pdf_files = get_pdfs_from_folder(folder_path)
                if "vectorstore" not in st.session_state:
                    if pdf_files:                    
                        with st.spinner("Traitement..."):
                            raw_text= extract_text_from_any_pdf(pdf_files)
                            text_chunks= get_text_chunks(raw_text)
                            vectorstore=st.session_state.vectorstore=get_vectorstore(text_chunks)
                            print("Vecteurs stores créés")
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                                
        if st.button("Gendarmerie la plus proche"):
            user_lat, user_lon = get_user_location()
           
            nearest_station, distance = find_nearest_station(user_lat, user_lon, gendarmeries)
            station_name, station_lat, station_lon = nearest_station

            # Générer le lien Google Maps
            google_maps_link = generate_google_maps_link(user_lat, user_lon, station_lat, station_lon)

            # Afficher les informations
            st.write(f"La gendarmerie la plus proche est : {station_name}")
            st.write(f"Distance : {distance} km")
            st.write(f"[Ouvrir dans Google Maps]({google_maps_link})")
           
        if st.button("Déposer une plainte"):
            plainte('A')
            
        st.success("🎯Nos objectifs")
        st.info("🤖ChatBot")
        st.warning("✨Assistance et orientation")
        st.info("👮‍♂️Sentez vous en sécurité")
        st.error("📑Déposez plainte")
     
    if st.session_state.vectorstore:                   
        input_container = st.container()
    
        with input_container:
        #display_chat()
        # Champ de saisie placé en bas
        
            st.markdown("""<div id="input-container" style='height: 20px;'></div>""", unsafe_allow_html=True)  # Espacement pour séparer les messages et l'entrée
            
            user_question= st.text_input("Posez votre question: ", value=st.session_state.user_question, key="input_field")
            if user_question:
                handle_userinput(user_question)
                display_chat()
                st.session_state.user_question = ""
                scroll_to_bottom()
                st.markdown(
                    """
                    <script>
                        document.getElementById('input-container').scrollIntoView({ behavior: 'smooth' });
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                
            #user_question=""
    
     
        
    user_lat, user_lon = get_user_location()
                            

if __name__ == '__main__':
    main()
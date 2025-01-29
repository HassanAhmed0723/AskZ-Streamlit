import streamlit as st
import time
import openai
import dotenv
import os
import json
import pandas as pd
import requests
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# Load the .env file from the parent directory
#dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Initialize OpenAI client
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = openai.OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_api_key = st.secrets["pinecone"]["PINECONE_API_KEY"]

# Log the secrets
st.write("Secrets:", st.secrets)

# Initialize embeddings
#embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=st.secrets["openai"]["OPENAI_API_KEY"])

# Initialize Pinecone Vector Store
vectorstore = PineconeVectorStore(
    index_name='askz-api-data',
    embedding=embeddings,
    text_key="full_content",
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4 
    }
)

def format_docs(docs):
    return "\n\n".join(f"Document from {doc.metadata.get('source_file', 'Unknown Source')}:\n{doc.page_content}" for doc in docs)


def execute_query(endpoint, method, parameters = None):
    url = "https://cip.stage.z360.biz" + endpoint
    headers = {
        'Authorization': 'Bearer 1',
        'Content-Type': 'application/json'
    }

    contact_data = None  # Initialize contact_data to avoid UnboundLocalError

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=parameters)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=parameters)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Check if the response is successful
        if response.status_code == 200:
            contact_data = response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            contact_data = {"error": f"API call failed with status code: {response.status_code}", "details": response.text}

    except requests.exceptions.RequestException as e:
        print(f"Error executing API call: {e}")
        contact_data = {"error": "RequestException", "details": str(e)}

    return contact_data


    # if method == "GET":
    #     response = requests.get(url, headers=headers, params=parameters)
    #     if response.status_code == 200:
    #         contact_data = response.json()
    #     else:
    #         print(f"Error: {response.status_code}, {response.text}")
    
    # return contact_data


def clean_response(response):
    # Remove any leading/trailing whitespace and ensure it's a valid JSON string
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    return response

def extract_json_data(api_endpoint_response):
    try:
        # Clean the response
        cleaned_response = clean_response(api_endpoint_response)
        # Extract JSON data from the response
        json_data = json.loads(cleaned_response)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    
def extract_api_call_details(api_data):
    try:
        # Clean the response
        cleaned_response = clean_response(api_data)
        
        # Load JSON data
        data = json.loads(cleaned_response)
        
        # Extract relevant details
        api_calls = data.get("api_calls", [])
        extracted_data = [
            {
                "endpoint": call.get("endpoint"),
                "method": call.get("method"),
                "parameters": call.get("parameters", {})
            }
            for call in api_calls
        ]
        
        return extracted_data
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def get_api_call_data(query):
    # Use the retriever to fetch relevant documents
    results = retriever.invoke(input=query) 

    # Check if results are empty
    if not results:
        return "I'm sorry, I couldn't find any relevant information in my knowledge base."

    # Format and return the documents
    formatted_docs = format_docs(results)
    print("Retrieved Chunks: \n",formatted_docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Prompt template to preprocess the user query
    answer_prompt = PromptTemplate.from_template(
            """ 
            Extract the most relevant API endpoint based on the provided user query and the API data Description.
            For user queries where a sequence of maximum two calls is required to get the results, extract all the API endpoints, methods and parameters.            
            Fill in the parameters provided by the user and only return the extracted Endpoints, Methods and Parameters (if explicitly provided by user) in JSON format like this:

            {{
                "api_calls": [
                    {{
                        "endpoint": "/api/endpoint1",
                        "method": "GET",
                        "parameters": {{
                            "param1": "value1",
                            "param2": "value2"
                        }}
                    }},
                    {{
                        "endpoint": "/api/endpoint2",
                        "method": "GET",
                        "parameters": {{
                            "paramA": "valueA",
                            "paramB": "valueB"
                        }}
                    }}
                ]
            }}
            
            User Query: {query}
            API Data: {formatted_docs}
            """
        )
    
    # Process the results
    api_data_chain = answer_prompt | llm | StrOutputParser()
    api_data = api_data_chain.invoke({"query": query, "formatted_docs": formatted_docs})

    print(f"API Request Data - Not Cleaned: {api_data}")
    # Clean and extract API call details
    api_call_details = extract_api_call_details(api_data)
    # Example output format
    print(api_call_details)

    # Save the first and second API calls
    first_api_call = api_call_details[0] if len(api_call_details) > 0 else None
    second_api_call = api_call_details[1] if len(api_call_details) > 1 else None

    # Check if the first API call exists
    if first_api_call:
        # Extract endpoint, method, and parameters
        endpoint = first_api_call.get("endpoint")
        method = first_api_call.get("method")
        parameters = first_api_call.get("parameters", {})

        # Execute the first API call
        first_call_response = execute_query(endpoint, method, parameters)
        print("First API Response:", first_call_response)
        first_call_response = json.dumps(first_call_response)
    else:
        print("No first API call details found.")
    
    print("Second API Call:", second_api_call)


    answer_prompt_2 = PromptTemplate.from_template(
        """
        Use the same endpoint and method as provided in the API Data and Just Fill in the required parameter for the API Call based on the provided user query, provided API Data and the results of the previously executed API Call.
        Return in JSON format like this:
            
            {{
                "endpoint":
                "method": 
                "parameters": {{
                    "param1": "value1",
                    "param2": "value2"
                }}
            }}
            
            User Query: {query}
            API Data: {second_api_call}
            First API Call Results: {first_call_response}

        """
    )

    # Process the results
    api_data_chain_2 = answer_prompt_2 | llm | StrOutputParser()
    api_data_2 = api_data_chain_2.invoke({"query": query, "second_api_call": second_api_call, "first_call_response": first_call_response})

    print(f"API Request Data - Not Cleaned: {api_data_2}")
    # Clean and extract API call details
    api_call_details_2 = extract_json_data(api_data_2)
    if api_call_details_2:
        extracted_endpoint = api_call_details_2.get("endpoint", "")
        extracted_method = api_call_details_2.get("method", "")
        extracted_parameters = api_call_details_2.get("parameters", {})
        second_call_response = execute_query(extracted_endpoint, extracted_method, extracted_parameters)
        print(f"Extracted Endpoint: {extracted_endpoint},\n Method: {extracted_method}, \n Parameters: {extracted_parameters}")
        print(f"Second Call Response: {second_call_response}")
        # print(f"Extracted JSON Data: {json_data}")
    else:
        print("Failed to extract JSON data.")

    #Convert the retrieved data to a string before submitting
    second_call_response = json.dumps(second_call_response)

    api_call_response = first_call_response + "\n" + second_call_response

    
    return f"{api_call_response}, \n\n Here's the data, provide only the most appropriate, relevant and user required information."


# Initialize Streamlit app
st.title("AskZ - Test")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread" not in st.session_state:
    st.session_state.thread = client.beta.threads.create()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Enter your message here:")
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=st.session_state.thread.id,
        role="user",
        content=user_query
    )

    # Run the Assistant with the new assistant ID
    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id="asst_hGF6GSizwpw58Mw2odcvqkLS",
    )

    assistant_reply = ""
    while True:
        time.sleep(2)
        run_status = client.beta.threads.runs.retrieve(
            thread_id=st.session_state.thread.id,
            run_id=run.id
        )

        if run_status.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread.id
            )

            for msg in messages.data:
                if msg.role == "assistant":
                    assistant_reply += msg.content[0].text.value
                    break

            if assistant_reply:
                with st.chat_message("assistant"):
                    st.markdown(assistant_reply)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

            break

        elif run_status.status == 'requires_action':
            required_actions = run_status.required_action.submit_tool_outputs.model_dump()
            tool_outputs = []

            for action in required_actions["tool_calls"]:
                func_name = action['function']['name']
                arguments = json.loads(action['function']['arguments'])

                if func_name == "get_api_call_data":
                    query = arguments["query"]
                    print(f"Query: {query}")
                    api_call_data = get_api_call_data(query)

                    tool_outputs.append({
                        "tool_call_id": action['id'],
                        "output": api_call_data
                    })

                else:
                    raise ValueError(f"Unknown function: {func_name}")

            client.beta.threads.runs.submit_tool_outputs(
                thread_id=st.session_state.thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        else:
            # st.text("Waiting for the Assistant to process...")
            time.sleep(2)

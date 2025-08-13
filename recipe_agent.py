# Install required Python packages
!pip install pandas
!pip install llama-index
!pip install openai
!pip install langchain
!pip install python-dotenv
!pip install transformers
!pip install langchain_community
!pip install llama_index.core

# Mount Google Drive to access the CSV file
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
import openai

# Set your OpenAI API key here
openai.api_key = "open-api-key"  # <-- Replace with your actual API key

# Function to load recipe data from a CSV file
def load_recipe_data(csv_path):
    """
    Reads a CSV file containing recipe data into a Pandas DataFrame.
    
    Args:
        csv_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: Loaded recipe data.
    """
    data = pd.read_csv(csv_path)
    return data

# Function to create document objects for the index
def create_documents(data):
    """
    Converts recipe DataFrame rows into LlamaIndex Document objects.
    
    Args:
        data (DataFrame): Recipe data.
    
    Returns:
        list[Document]: List of Document objects for indexing.
    """
    documents = []
    for _, row in data.iterrows():
        # Create a text representation of each recipe
        document_text = (
            f"Title: {row['Name']}\n"
            f"RecipeIngredientParts: {row['RecipeIngredientParts']}\n"
            f"Keywords: {row['Keywords']}\n"
            f"TotalTime: {row['TotalTime']}\n"
            f"Calories: {row['Calories']}\n"
            f"Category: {row['RecipeCategory']}\n"
        )
        documents.append(Document(text=document_text))
    return documents

# Function to create and persist a recipe index
def create_recipe_index(documents, index_directory="recipe_index"):
    """
    Creates a VectorStoreIndex from recipe documents and saves it locally.
    
    Args:
        documents (list[Document]): List of recipe documents.
        index_directory (str): Directory to store the index.
    
    Returns:
        VectorStoreIndex: Created recipe index.
    """
    import os
    os.makedirs(index_directory, exist_ok=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_directory)
    print("create_recipe_index success")
    return index

# Function to set up the recipe agent
def setup_recipe_agent(index_directory):
    """
    Loads a saved recipe index and initializes an AI agent to query it.
    
    Args:
        index_directory (str): Path to stored index files.
    
    Returns:
        AgentExecutor: Initialized agent for querying recipes.
    """
    try:
        # Load stored index
        storage_context = StorageContext.from_defaults(persist_dir=index_directory)
        index = load_index_from_storage(storage_context)
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

    # Create a ChatOpenAI LLM instance
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai.api_key)

    # Define a tool for recipe search
    tools = [
        Tool(
            name="Recipe Generator",
            func=index.as_query_engine().query,
            description="Use this tool to search recipes based on ingredients, category, time, or keywords."
        )
    ]

    # Initialize an agent with zero-shot reasoning
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    print("setup_recipe_agent success")
    return agent

# Function to query the recipe agent
def query_recipe_agent(agent, user_query):
    """
    Sends a query to the recipe agent and retrieves results.
    
    Args:
        agent (AgentExecutor): The initialized recipe agent.
        user_query (str): User's search query.
    
    Returns:
        str: Agent's response with recipe suggestions.
    """
    response = agent.run(user_query)
    print("query_recipe_agent success")
    return response

# Main execution block
if __name__ == "__main__":
    # Path to the recipes CSV file in Google Drive
    csv_file_path = '/content/drive/MyDrive/recipes.csv'

    # Step 1: Load recipe data
    recipe_data = load_recipe_data(csv_file_path)

    # Step 2: Convert recipe data into documents
    documents = create_documents(recipe_data)

    # Step 3: Create and persist recipe index
    index_filename = "recipe_index"
    create_recipe_index(documents, index_filename)

    # Step 4: Set up the recipe agent
    agent = setup_recipe_agent(index_filename)

    # Step 5: Accept user query
    print("Enter your recipe preferences (e.g., Name, Total cook Time, Calories, Ingredients or keywords like 'Dessert', 'Vegan'):")
    user_query = input(" ")

    # Step 6: Query the agent and display results
    Recipes = query_recipe_agent(agent, user_query)
    print("\nGenerated Recipes:")
    print(Recipes)